# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Size cap for the filesystem tier: LRU eviction of block files.

The fs tier writes every block it is given and never deletes a file, so a
long-running server fills its disk. FsTierEvictor bounds the bytes of block
files in the engine's block directory (<base>_r<rank>):

    touch(path)              a load, or a store that finds the file on disk:
                             move the block to the most recently used end
    reserve(path, nbytes)    before a write: evict least recently used files
                             until the write fits; False skips the block
    commit(path)/abort(path) after the write landed / failed

A full cap evicts down to ``eviction_ratio * max_bytes``. ``min_free_bytes``
also keeps that much of the filesystem free, counting writes in flight.
Writes in flight are never evicted. Files from an earlier start join the index
at construction, oldest first. The engine must be the only writer of its block
directory: with data parallel > 1, give each replica its own root_dir.
"""

import os
import threading
import time
from collections import OrderedDict

from vllm.logger import init_logger

logger = init_logger(__name__)

_LOG_INTERVAL_S = 60.0


class FsTierEvictor:
    def __init__(
        self,
        block_dir: str,
        max_bytes: int = 0,
        min_free_bytes: int = 0,
        eviction_ratio: float = 0.9,
    ) -> None:
        if max_bytes < 0 or min_free_bytes < 0:
            raise ValueError("max_bytes and min_free_bytes must be >= 0")
        if not 0.0 < eviction_ratio <= 1.0:
            raise ValueError(f"eviction_ratio must be in (0, 1], got {eviction_ratio}")
        self.block_dir = block_dir
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self.eviction_ratio = eviction_ratio
        self._lru: OrderedDict[str, int] = OrderedDict()  # path -> bytes, LRU first
        self._pending: dict[str, int] = {}  # reserved, write not landed yet
        self._pending_bytes = 0
        self._used = 0
        self._lock = threading.Lock()
        self._evicted_bytes = 0
        self._refused = 0
        self._last_log = time.monotonic()

        os.makedirs(block_dir, exist_ok=True)
        self._scan()
        with self._lock:
            if self.max_bytes:
                self._evict_down_to(self.max_bytes)
        logger.info(
            "fs tier: cap %d B (evicts to %.2f), free floor %d B, "
            "indexed %d existing block files (%d B)",
            self.max_bytes,
            self.eviction_ratio,
            self.min_free_bytes,
            len(self._lru),
            self._used,
        )

    def touch(self, path: str, wait: bool = True) -> bool:
        """Move an indexed block to the most recently used end. False if the
        index does not hold it. ``wait=False`` skips the move while an
        eviction pass holds the lock (loads must not stall behind unlinks)."""
        if not self._lock.acquire(blocking=wait):
            return True
        try:
            if path not in self._lru:
                return False
            self._lru.move_to_end(path)
            return True
        finally:
            self._lock.release()

    def reserve(self, path: str, nbytes: int) -> bool:
        with self._lock:
            if self.max_bytes:
                if nbytes > self.max_bytes:
                    return self._refuse()
                if self._used + nbytes > self.max_bytes:
                    target = int(self.max_bytes * self.eviction_ratio) - nbytes
                    self._evict_down_to(target)
                    if self._used + nbytes > self.max_bytes:
                        return self._refuse()
            if self.min_free_bytes:
                need = self.min_free_bytes + self._pending_bytes + nbytes
                free = self._free()
                if free is not None and free < need:
                    # Each unlink frees about its own size on a local disk.
                    self._evict_down_to(self._used - (need - free))
                    free = self._free()
                    if free is not None and free < need:
                        return self._refuse()
            self._used += nbytes - self._lru.pop(path, 0)
            self._lru[path] = nbytes
            self._pending[path] = nbytes
            self._pending_bytes += nbytes
            self._maybe_log()
            return True

    def commit(self, path: str) -> None:
        with self._lock:
            self._pending_bytes -= self._pending.pop(path, 0)

    def abort(self, path: str) -> None:
        with self._lock:
            self._pending_bytes -= self._pending.pop(path, 0)
            self._used -= self._lru.pop(path, 0)

    def _evict_down_to(self, target: int) -> None:
        """Unlink least recently used files until used <= target. Holds _lock.

        ponytail: unlinks run under the lock, as in SGLang's file evictor; a
        full cap frees (1 - eviction_ratio) x max_bytes in one pass, so stores
        wait that long (loads skip their touch instead). Raise eviction_ratio
        if the pause shows up in store latency.
        """
        skipped = []
        start, freed, files = time.monotonic(), self._used, 0
        while self._used > target and self._lru:
            path, size = self._lru.popitem(last=False)
            if path in self._pending:
                skipped.append((path, size))
                continue
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                # Dropped from the index rather than retried on every pass.
                logger.warning("fs tier: cannot evict %s: %s", path, exc)
            self._used -= size
            self._evicted_bytes += size
            files += 1
        for path, size in reversed(skipped):  # back at the LRU end, in order
            self._lru[path] = size
            self._lru.move_to_end(path, last=False)
        elapsed = time.monotonic() - start
        if elapsed >= 1.0:
            logger.info(
                "fs tier: evicted %d files (%d B) in %.1f s",
                files,
                freed - self._used,
                elapsed,
            )

    def _refuse(self) -> bool:
        self._refused += 1
        self._maybe_log()
        return False

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log < _LOG_INTERVAL_S:
            return
        self._last_log = now
        (logger.warning if self._refused else logger.info)(
            "fs tier: %d B in %d files; last minute: %d B evicted, "
            "%d blocks not stored",
            self._used,
            len(self._lru),
            self._evicted_bytes,
            self._refused,
        )
        self._evicted_bytes = self._refused = 0

    def _free(self) -> int | None:
        try:
            st = os.statvfs(self.block_dir)
        except OSError:
            return None
        return st.f_bavail * st.f_frsize

    def _scan(self) -> None:
        """Index block files left by an earlier start, oldest mtime first, and
        drop temp files (<hash>.bin_<n>.tmp) of writes that never finished."""
        found = []
        for dirpath, _, names in os.walk(self.block_dir):
            for name in names:
                path = os.path.join(dirpath, name)
                try:
                    if name.endswith(".tmp") and ".bin_" in name:
                        os.remove(path)
                    elif name.endswith(".bin"):
                        st = os.stat(path)
                        found.append((st.st_mtime, path, st.st_size))
                except OSError:
                    continue
        for _, path, size in sorted(found):
            self._lru[path] = size
            self._used += size
