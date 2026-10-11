# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming DeepSeek-V4.1 with the Run:ai streamer, on CPU.

Each engram table is one tensor larger than a GPU, so the streamer delivers it
in row blocks and every tensor-parallel rank keeps only its own rows. The tests
drive the real streamer, the real iterator, and the real DeepSeek-V4.1 wrapper,
text model and engram loaders; only GPU-only calls and the TP rank are faked.
"""

import os
import weakref
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.runai_streamer_loader import (
    RunaiModelStreamerLoader,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.deepseek_v41.common import engram as engram_mod
from vllm.models.deepseek_v41.nvidia import model as v41_model
from vllm.models.deepseek_v41.nvidia import vl_model as v41_vl

pytest.importorskip("runai_model_streamer")

ROW_BLOCKS = v41_vl.DeepseekV41ForCausalLM.row_block_weights
BLOCK_BYTES = 10_000  # splits the 256-byte-row table into blocks of 39 rows
HEAD_DIM = 256


def _head_sizes(layer: int) -> tuple[int, ...]:
    # The real layout (4-grams, 8 heads, so 24 prime-sized hash columns) over
    # a small vocabulary, so the TP ranks own uneven row ranges.
    config = SimpleNamespace(
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[0, 0],
        engram_max_ngram_size=4,
        engram_n_heads=8,
        engram_head_dim=HEAD_DIM,
        engram_compressed_vocab_size=0,
        engram_pad_token_id=2,
        engram_vocab_size=40,
    )
    primes = engram_mod.EngramLayout(config).primes[layer]
    return tuple(size for per_ngram in primes for size in per_ngram)


def _fake_rank(rank: int, world: int, set_attr=setattr) -> None:
    """Pose as TP rank `rank` of `world`, and stub the GPU lookups the engram
    module makes at construction."""
    for module in (engram_mod, v41_model):
        set_attr(module, "get_tensor_model_parallel_rank", lambda: rank)
        set_attr(module, "get_tensor_model_parallel_world_size", lambda: world)
    set_attr(
        torch.cuda,
        "get_device_properties",
        lambda *_: SimpleNamespace(multi_processor_count=1),
    )
    set_attr(torch.accelerator, "current_device_index", lambda: 0)


def _random(shape, dtype) -> torch.Tensor:
    return torch.randint(0, 120, shape, dtype=torch.uint8).view(dtype)


def _write_checkpoint(path) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Three shards, so "language_model." weights arrive in two runs split by
    vision weights, as the real checkpoint streams them."""
    rows = sum(_head_sizes(0))
    shards = [
        {
            "norm.weight": torch.randn(16),
            "layers.1.engram.embed.weight": _random(
                (rows, HEAD_DIM), torch.float8_e4m3fn
            ),
            "layers.1.engram.embed.scale": _random(
                (rows, HEAD_DIM // 32), torch.float8_e8m0fnu
            ),
        },
        {"vision.proj.weight": torch.randn(8, 8)},
        # Larger than a block but not opted in: must arrive whole.
        {"layers.1.engram.wkv.weight": _random((64, HEAD_DIM), torch.float8_e4m3fn)},
    ]
    files = []
    for i, tensors in enumerate(shards):
        files.append(str(path / f"model-{i:05d}.safetensors"))
        save_file(tensors, files[-1])
    return {name: t for shard in shards for name, t in shard.items()}, files


class _TextModel(nn.Module):
    """Stands in for DeepseekV4Model: its real load_weights, a minimal tree."""

    load_weights = v41_model.DeepseekV4Model.load_weights

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=64)
        self.quant_config = None
        self.use_sequence_parallel = False
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.empty(16), requires_grad=False)
        engram = nn.Module()
        heads = _head_sizes(0)
        engram.embed_tokens = engram_mod.ParallelEngramEmbedding(
            sum(heads), HEAD_DIM, heads
        )
        engram.wkv = nn.Module()
        engram.wkv.weight = nn.Parameter(
            torch.empty(64, HEAD_DIM, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.layers = nn.ModuleList([nn.Module(), nn.Module()])
        self.layers[1].engram = engram
        self.finalized_after = []

    def get_expert_mapping(self):
        return []

    def finalize_mega_moe_weights(self) -> None:
        self.finalized_after.append(self.loaded_count)

    def finalize_mhc_broadcast_weights(self) -> None:
        pass


class _LanguageModel(nn.Module):
    load_weights = v41_model.DeepseekV41LLMForCausalLM.load_weights
    process_weights_after_loading = (
        v41_model.DeepseekV41LLMForCausalLM.process_weights_after_loading
    )

    def __init__(self) -> None:
        super().__init__()
        self.model = _TextModel()
        self.hf_to_vllm_mapper = WeightsMapper()


class _Wrapper(nn.Module):
    """The real DeepseekV41ForCausalLM loading methods on a minimal tree."""

    row_block_weights = ROW_BLOCKS

    def __init__(self) -> None:
        super().__init__()
        self.language_model = _LanguageModel()
        self.vision = nn.Module()
        self.vision.proj = nn.Module()
        self.vision.proj.weight = nn.Parameter(torch.empty(8, 8), requires_grad=False)
        self.hf_to_vllm_mapper = v41_vl._make_deepseek_v4_vl_weights_mapper(
            "fp4", "weight_scale"
        )

    def load_weights(self, weights):
        """The real wrapper load_weights, recording what it is fed: the most
        weight tensors alive at once (all of them if it holds the stream) and
        each item's row_offset."""
        text = self.language_model.model
        live = []

        def tracked():
            for name, tensor in weights:
                live.append(weakref.ref(tensor))
                self.most_live = max(
                    self.most_live, sum(ref() is not None for ref in live)
                )
                self.row_offsets.append(getattr(tensor, "row_offset", None))
                yield name, tensor
                text.loaded_count += 1

        return v41_vl.DeepseekV41ForCausalLM.load_weights(self, tracked())


def _load(load) -> _Wrapper:
    """`load(model)` must load the model once; checks it finalized exactly
    once, after every streamed item was loaded."""
    model = _Wrapper()
    model.most_live, model.row_offsets = 0, []
    text = model.language_model.model
    text.loaded_count = 0
    load(model)
    assert text.finalized_after == [text.loaded_count], text.finalized_after
    return model


def _runai_loader(root: str, **extra_config):
    """The real Run:ai streamer loader, as `vllm serve <root>` builds it."""
    loader = RunaiModelStreamerLoader(
        LoadConfig(load_format="runai_streamer", model_loader_extra_config=extra_config)
    )
    config = SimpleNamespace(model=root, model_weights=None, revision=None)
    return lambda model: loader.load_weights(model, config)


def _check_loaded(model: _Wrapper, expected: dict[str, torch.Tensor]) -> None:
    text = model.language_model.model
    embed = text.layers[1].engram.embed_tokens
    rows = slice(embed.vocab_start_idx, embed.vocab_end_idx)
    table = expected["layers.1.engram.embed.weight"].view(torch.uint8)
    scale = expected["layers.1.engram.embed.scale"].view(torch.uint8)
    assert torch.equal(embed.weight.view(torch.uint8), table[rows])
    assert torch.equal(embed.weight_scale_inv, scale[rows])
    assert torch.equal(text.norm.weight, expected["norm.weight"])
    assert torch.equal(
        text.layers[1].engram.wkv.weight.view(torch.uint8),
        expected["layers.1.engram.wkv.weight"].view(torch.uint8),
    )
    assert torch.equal(model.vision.proj.weight, expected["vision.proj.weight"])


def test_split_streams_every_byte_once(tmp_path, monkeypatch):
    monkeypatch.setattr(weight_utils, "RUNAI_ROW_BLOCK_BYTES", BLOCK_BYTES)
    expected, files = _write_checkpoint(tmp_path)
    prepare_request = weight_utils.safetensors_pytorch.prepare_request

    blocks: dict[str, list[tuple[int, torch.Tensor]]] = {}
    for name, tensor in weight_utils.runai_safetensors_weights_iterator(
        files, False, row_block_weights=ROW_BLOCKS
    ):
        blocks.setdefault(name, []).append(
            (getattr(tensor, "row_offset", None), tensor)
        )

    assert weight_utils.safetensors_pytorch.prepare_request is prepare_request
    assert blocks.keys() == expected.keys()
    for name, parts in blocks.items():
        if name.endswith(ROW_BLOCKS):
            parts.sort(key=lambda part: part[0])
            assert len(parts) > 1
            assert all(t.numel() * t.element_size() <= BLOCK_BYTES for _, t in parts)
            starts = [start for start, _ in parts]
            ends = [start + t.shape[0] for start, t in parts]
            assert starts == [0, *ends[:-1]] and ends[-1] == expected[name].shape[0]
            whole = torch.cat([t.view(torch.uint8) for _, t in parts])
        else:
            [(start, whole)] = parts
            assert start is None
            whole = whole.view(torch.uint8)
        assert torch.equal(whole, expected[name].view(torch.uint8))


@pytest.mark.parametrize("tp_size", [1, 4, 8])
def test_engram_loader_blocks_match_whole_table(monkeypatch, tp_size):
    heads = _head_sizes(1)
    table = _random((sum(heads), HEAD_DIM), torch.float8_e4m3fn)
    for rank in range(tp_size):
        _fake_rank(rank, tp_size, monkeypatch.setattr)
        whole = engram_mod.ParallelEngramEmbedding(sum(heads), HEAD_DIM, heads)
        blocked = engram_mod.ParallelEngramEmbedding(sum(heads), HEAD_DIM, heads)
        whole.weight.weight_loader(whole.weight, table)
        for start in range(0, table.shape[0], 37):
            block = table[start : start + 37].clone()
            block.row_offset = start
            blocked.weight.weight_loader(blocked.weight, block)
        rows = slice(whole.vocab_start_idx, whole.vocab_end_idx)
        expected = table.view(torch.uint8)[rows]
        assert torch.equal(whole.weight.view(torch.uint8), expected)
        assert torch.equal(blocked.weight.view(torch.uint8), expected)


def _check_streamed(model: _Wrapper) -> None:
    # The engram tables arrived in row blocks, and the wrapper held at most a
    # few streamed tensors at a time.
    assert sum(offset is not None for offset in model.row_offsets) > 2
    assert model.most_live <= 3, model.most_live


def test_wrapper_loads_lazily_and_finalizes_once(tmp_path, monkeypatch):
    monkeypatch.setattr(weight_utils, "RUNAI_ROW_BLOCK_BYTES", BLOCK_BYTES)
    expected, files = _write_checkpoint(tmp_path)
    _fake_rank(1, 4, monkeypatch.setattr)
    model = _load(_runai_loader(str(tmp_path)))
    _check_loaded(model, expected)
    _check_streamed(model)
    # The local-volume path (mmap views, no row blocks) loads the same model.
    model = _load(
        lambda m: m.load_weights(
            weight_utils.safetensors_weights_iterator(files, False)
        )
    )
    _check_loaded(model, expected)


def _distributed_rank(rank: int, world: int, root: str) -> None:
    os.environ["RUNAI_STREAMER_DIST"] = "1"  # allow the gloo backend on CPU
    os.environ["RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE"] = str(1 << 20)
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{root}/pg", rank=rank, world_size=world
    )
    weight_utils.RUNAI_ROW_BLOCK_BYTES = BLOCK_BYTES
    _fake_rank(rank, world)
    used_distributed = []

    class Streamer(weight_utils.SafetensorsStreamer):
        def stream_files(self, *args, **kwargs):
            super().stream_files(*args, **kwargs)
            used_distributed.append(self.file_streamer.is_distributed)

    weight_utils.SafetensorsStreamer = Streamer
    expected = torch.load(os.path.join(root, "expected.pt"))
    model = _load(_runai_loader(root, distributed=True))
    assert used_distributed[-1], "the streamer fell back to non-distributed reads"
    _check_loaded(model, expected)
    _check_streamed(model)
    torch.distributed.destroy_process_group()


def test_distributed_streaming_loads_each_rank(tmp_path):
    expected, _ = _write_checkpoint(tmp_path)
    torch.save(expected, tmp_path / "expected.pt")
    torch.multiprocessing.spawn(
        _distributed_rank, args=(4, str(tmp_path)), nprocs=4, join=True
    )
