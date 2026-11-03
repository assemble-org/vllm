# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Size cap (max_bytes / min_free_bytes) of the filesystem tier."""

import os
import threading

import pytest
from vllm.v1.kv_offload.tiering.fs import io as fs_io
from vllm.v1.kv_offload.tiering.fs import manager as fs_manager
from vllm.v1.kv_offload.tiering.fs.evictor import FsTierEvictor
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

from tests.v1.kv_offload.tiering.test_fs_tier import (
    _BLOCK_ELEMENTS,
    _MOCK_OFFLOADING_SPEC,
    _NUM_BLOCKS,
    _page_aligned_zero_tensor,
    drain,
    key,
    make_job,
)


def make_tier(root, blocks_in_cap=None, **kwargs) -> FileSystemTierManager:
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    block = memoryview(tensor.numpy()).strides[0]
    if blocks_in_cap is not None:
        kwargs["max_bytes"] = blocks_in_cap * block
    return FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(root),
        n_read_threads=2,
        n_write_threads=2,
        **kwargs,
    )


def store(tier, job_id, ns):
    tier.submit_store(make_job(job_id, [key(n) for n in ns], [0] * len(ns)))
    return drain(tier)


def on_disk(tier, ns):
    return [os.path.exists(tier.file_mapper.get_file_name(key(n))) for n in ns]


def test_full_cap_evicts_lru_down_to_ratio(tmp_path):
    tier = make_tier(tmp_path, blocks_in_cap=4, eviction_ratio=0.5)
    try:
        store(tier, 1, [1, 2, 3, 4])
        assert on_disk(tier, [1, 2, 3, 4]) == [True] * 4
        store(tier, 2, [5])  # 5 blocks > 4: evict to 2 - 1, then write
        assert on_disk(tier, [1, 2, 3, 4, 5]) == [False, False, False, True, True]
        assert tier._evictor._used == 2 * tier._block_size
    finally:
        tier.shutdown()


def test_load_moves_block_to_front(tmp_path):
    tier = make_tier(tmp_path, blocks_in_cap=3, eviction_ratio=1.0)
    try:
        store(tier, 1, [1, 2, 3])
        tier.submit_load(make_job(2, [key(1)], [1], is_promotion=True))
        assert all(r.success for r in drain(tier))
        store(tier, 3, [4])  # evicts 2, the least recently used, not 1
        assert on_disk(tier, [1, 2, 3, 4]) == [True, False, True, True]
    finally:
        tier.shutdown()


def test_store_of_existing_file_moves_it_and_counts_once(tmp_path):
    tier = make_tier(tmp_path, blocks_in_cap=3, eviction_ratio=1.0)
    try:
        store(tier, 1, [1, 2, 3])
        # Already on disk, at a full cap: touched, nothing evicted or rewritten.
        store(tier, 2, [2])
        assert on_disk(tier, [1, 2, 3]) == [True, True, True]
        assert tier._evictor._used == 3 * tier._block_size
        store(tier, 3, [4])
        assert on_disk(tier, [1, 2, 3, 4]) == [False, True, True, True]
    finally:
        tier.shutdown()


def test_store_touches_its_found_blocks_before_reserving(tmp_path, monkeypatch):
    # The C lookup answers for the whole job up front, so a reserve for a new
    # block must not evict a block of the same job that is already on disk.
    monkeypatch.setattr(
        fs_manager, "_exists", lambda paths: [os.path.exists(p) for p in paths]
    )
    tier = make_tier(tmp_path, blocks_in_cap=2, eviction_ratio=1.0)
    try:
        store(tier, 1, [1, 2])
        first = tier.file_mapper.get_file_name(key(1))
        os.utime(first, (1e9, 1e9))
        store(tier, 2, [3, 1])
        assert on_disk(tier, [1, 2, 3]) == [True, False, True]
        assert os.path.getmtime(first) == 1e9  # kept, not evicted and rewritten
    finally:
        tier.shutdown()


def test_free_space_floor_skips_the_block(tmp_path, monkeypatch):
    monkeypatch.setattr(FsTierEvictor, "_free", lambda self: 0)
    tier = make_tier(tmp_path, min_free_bytes=1)
    try:
        assert [r.success for r in store(tier, 1, [1])] == [True]
        assert on_disk(tier, [1]) == [False]
        assert tier._evictor._used == 0
    finally:
        tier.shutdown()


def test_failed_batch_keeps_only_written_files(tmp_path, monkeypatch):
    tier = make_tier(tmp_path, blocks_in_cap=8)
    real = fs_io._store_block
    calls = []

    def fail_second(path, *args, **kwargs):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("disk full")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(fs_io, "_HAS_FSIO_C", False)
    monkeypatch.setattr(fs_io, "_store_block", fail_second)
    try:
        assert [r.success for r in store(tier, 1, [1, 2, 3])] == [False]
        assert on_disk(tier, [1, 2, 3]) == [True, False, False]
        assert tier._evictor._used == tier._block_size
        assert not tier._evictor._pending
    finally:
        tier.shutdown()


def test_any_store_error_releases_its_reservations(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("bad offsets")

    monkeypatch.setattr(fs_manager, "batch_store_block", broken)
    tier = make_tier(tmp_path, blocks_in_cap=8)
    try:
        assert [r.success for r in store(tier, 1, [1, 2])] == [False]
        assert tier._evictor._used == 0
        assert not tier._evictor._pending
    finally:
        tier.shutdown()


def test_restart_indexes_old_files_oldest_first(tmp_path):
    tier = make_tier(tmp_path)
    store(tier, 1, [1, 2, 3])
    paths = [tier.file_mapper.get_file_name(key(n)) for n in (1, 2, 3)]
    tier.shutdown()
    for age, path in zip((300, 100, 200), paths):  # 1 oldest, then 3, then 2
        os.utime(path, (1e9 - age, 1e9 - age))
    stale_tmp = paths[0] + "_123.tmp"
    open(stale_tmp, "wb").close()
    # Files outside the engine's block directory are not the tier's.
    others = [tmp_path / "weights.bin", tmp_path / "upload.tmp"]
    for other in others:
        other.write_bytes(b"\0")
        os.utime(other, (1, 1))

    tier = make_tier(tmp_path, blocks_in_cap=2, eviction_ratio=1.0)
    try:  # start-up trims 3 old files to the cap of 2: the oldest goes
        assert on_disk(tier, [1, 2, 3]) == [False, True, True]
        assert not os.path.exists(stale_tmp)
        assert all(os.path.exists(other) for other in others)
        store(tier, 2, [4])  # then 3, older than 2
        assert on_disk(tier, [2, 3, 4]) == [True, False, True]
    finally:
        tier.shutdown()


def write(path, nbytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * nbytes)


def test_pending_writes_are_never_evicted(tmp_path):
    ev = FsTierEvictor(str(tmp_path), max_bytes=20, eviction_ratio=1.0)
    a, b = str(tmp_path / "a.bin"), str(tmp_path / "b.bin")
    assert ev.reserve(a, 10) and ev.reserve(b, 10)
    assert not ev.reserve(str(tmp_path / "c.bin"), 10)  # both still in flight
    write(a, 10)
    ev.commit(a)
    assert ev.reserve(str(tmp_path / "c.bin"), 10)  # now a can go
    assert not os.path.exists(a)


def test_free_space_floor_counts_writes_in_flight(tmp_path, monkeypatch):
    monkeypatch.setattr(FsTierEvictor, "_free", lambda self: 15)
    ev = FsTierEvictor(str(tmp_path), min_free_bytes=1)
    a, b = str(tmp_path / "a.bin"), str(tmp_path / "b.bin")
    assert ev.reserve(a, 10)
    assert not ev.reserve(b, 10)  # 15 free, but 10 already promised to a
    ev.abort(a)
    assert ev.reserve(b, 10)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can unlink anything")
def test_undeletable_file_does_not_stop_eviction(tmp_path):
    stuck = tmp_path / "ro" / "a.bin"
    write(str(stuck), 10)
    os.utime(stuck, (1, 1))  # the least recently used file
    write(str(tmp_path / "b.bin"), 10)
    os.chmod(stuck.parent, 0o500)
    try:
        ev = FsTierEvictor(str(tmp_path), max_bytes=20, eviction_ratio=1.0)
        for name in ("c.bin", "d.bin"):
            path = str(tmp_path / name)
            assert ev.reserve(path, 10)
            write(path, 10)
            ev.commit(path)
    finally:
        os.chmod(stuck.parent, 0o700)


def test_concurrent_reserves_stay_under_the_cap(tmp_path):
    ev = FsTierEvictor(str(tmp_path), max_bytes=100, eviction_ratio=0.8)
    over = []

    def worker(t):
        for i in range(200):
            path = str(tmp_path / f"{t}_{i}.bin")
            if ev.reserve(path, 10):
                write(path, 10)
                ev.commit(path)
                if ev._used > 100:
                    over.append(ev._used)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not over
    files = [f for f in os.listdir(tmp_path) if f.endswith(".bin")]
    assert len(files) * 10 == ev._used <= 100
