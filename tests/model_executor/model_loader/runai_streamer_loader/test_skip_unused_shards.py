# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Run:ai streamer loader streams only the shards a draft model loads.

An MTP head lives in a few shards of its target's checkpoint. The streamer reads
every byte of the files it is given, so skipping the head's unused tensors in
`load_weights` comes too late: the loader must drop those shards before
streaming. The tests drive the real streamer and loader on CPU.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import runai_streamer_loader, weight_utils
from vllm.model_executor.model_loader.runai_streamer_loader import (
    RunaiModelStreamerLoader,
)

pytest.importorskip("runai_model_streamer")


def _tensor(seed: int, *shape: int) -> torch.Tensor:
    # Deterministic, so spawned ranks rebuild the same values.
    numel = torch.Size(shape).numel()
    return torch.arange(seed, seed + numel, dtype=torch.float32).reshape(shape)


def _shards() -> list[dict[str, torch.Tensor]]:
    # Shard 3 holds only draft (layer 2) weights; shard 4 mixes both.
    return [
        {
            "model.embed_tokens.weight": _tensor(0, 32, 8),
            "model.layers.0.w": _tensor(1, 8, 8),
        },
        {"model.layers.1.w": _tensor(2, 8, 8)},
        {"model.layers.2.w": _tensor(3, 8, 8)},
        {"model.layers.1.v": _tensor(4, 8), "model.layers.2.v": _tensor(5, 8)},
    ]


def _write_checkpoint(root: str) -> list[str]:
    files, weight_map = [], {}
    shards = _shards()
    for i, tensors in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        files.append(os.path.join(root, name))
        save_file(tensors, files[-1])
        weight_map.update(dict.fromkeys(tensors, name))
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    return files


class _Target(nn.Module):
    """Records every tensor it is fed."""

    def __init__(self) -> None:
        super().__init__()
        self.fed: dict[str, torch.Tensor] = {}

    def load_weights(self, weights) -> None:
        self.fed.update(weights)


class _Draft(_Target):
    """An MTP head on a two-layer target: it loads layer 2 only."""

    def is_unused_checkpoint_weight(self, name: str) -> bool:
        return not name.startswith("model.layers.2.")


def _fed_from(shard_ids: list[int]) -> dict[str, torch.Tensor]:
    shards = _shards()
    return {name: t for i in shard_ids for name, t in shards[i].items()}


def _check_fed(model: _Target, expected: dict[str, torch.Tensor]) -> None:
    assert model.fed.keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(model.fed[name], tensor), name


def _record_streamed(set_attr) -> list[tuple[list[str], bool]]:
    """Make the iterator's streamer record each call's files and whether it
    streamed them in distributed mode."""
    streamed = []

    class Streamer(weight_utils.SafetensorsStreamer):
        def stream_files(self, paths, *args, **kwargs):
            super().stream_files(paths, *args, **kwargs)
            streamed.append((sorted(paths), self.file_streamer.is_distributed))

    set_attr(weight_utils, "SafetensorsStreamer", Streamer)
    return streamed


def _load(model: nn.Module, root: str, model_weights: str = "", **extra) -> None:
    loader = RunaiModelStreamerLoader(
        LoadConfig(load_format="runai_streamer", model_loader_extra_config=extra)
    )
    config = SimpleNamespace(model=root, model_weights=model_weights, revision=None)
    loader.load_weights(model, config)


def test_draft_streams_only_the_shards_it_loads(tmp_path, monkeypatch):
    files = _write_checkpoint(str(tmp_path))
    streamed = _record_streamed(monkeypatch.setattr)

    draft = _Draft()
    _load(draft, str(tmp_path))
    assert [paths for paths, _ in streamed] == [files[2:]]
    _check_fed(draft, _fed_from([2, 3]))

    # A model that names no unused weights still streams every shard.
    target = _Target()
    _load(target, str(tmp_path))
    assert streamed[-1][0] == files
    _check_fed(target, _fed_from([0, 1, 2, 3]))


def test_object_storage_reads_the_index_pulled_with_the_config(tmp_path, monkeypatch):
    # With weights in object storage, `model` is the local directory vLLM pulled
    # the config and index into, and `model_weights` is the bucket.
    files = _write_checkpoint(str(tmp_path))
    bucket = "s3://bucket/checkpoint"
    in_bucket = [f"{bucket}/{os.path.basename(f)}" for f in files]
    monkeypatch.setattr(
        runai_streamer_loader, "list_safetensors", lambda path: in_bucket
    )
    streamed = []
    monkeypatch.setattr(
        runai_streamer_loader,
        "runai_safetensors_weights_iterator",
        lambda paths, *args: streamed.append(paths) or iter(()),
    )
    _load(_Draft(), str(tmp_path), model_weights=bucket)
    assert streamed == [in_bucket[2:]]


def test_index_filter_keeps_shards_it_cannot_vouch_for(tmp_path):
    files = _write_checkpoint(str(tmp_path))
    index = str(tmp_path / "model.safetensors.index.json")
    is_unused = _Draft().is_unused_checkpoint_weight
    keep = weight_utils.filter_safetensors_files_by_index
    unlisted = str(tmp_path / "consolidated.safetensors")
    assert keep(files, index, is_unused) == files[2:]
    assert keep([*files, unlisted], index, is_unused) == [*files[2:], unlisted]
    assert keep(files, str(tmp_path / "missing.json"), is_unused) == files
    assert keep(files, index, lambda name: True) == files


def _distributed_rank(rank: int, world: int, root: str) -> None:
    os.environ["RUNAI_STREAMER_DIST"] = "1"  # allow the gloo backend on CPU
    os.environ["RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE"] = str(1 << 20)
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{root}/pg", rank=rank, world_size=world
    )
    # gloo streams to CPU only; on a GPU host the iterator would pick cuda:N.
    weight_utils.current_platform.is_cuda_alike = lambda: False
    streamed = _record_streamed(setattr)
    draft = _Draft()
    _load(draft, root, distributed=True)
    files = sorted(
        os.path.join(root, f) for f in os.listdir(root) if f.endswith(".safetensors")
    )
    assert streamed == [(files[2:], True)], streamed
    _check_fed(draft, _fed_from([2, 3]))
    torch.distributed.destroy_process_group()


def test_distributed_draft_streams_fewer_shards_than_ranks(tmp_path):
    _write_checkpoint(str(tmp_path))
    torch.multiprocessing.spawn(
        _distributed_rank, args=(4, str(tmp_path)), nprocs=4, join=True
    )
