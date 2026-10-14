# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 indexer ``wk`` (weight, weight_scale_inv) pairs in the DSA loader
(DeepSeek-V3.2, GLM-5.3) must survive being delivered in different
``load_weights`` calls.

``DeepseekV32Model.load_weights`` dequantizes the FP8 ``indexer.wk`` weight to
BF16 for its fusion into ``wk_weights_proj`` once both halves of the pair have
been seen, parking the first half in a pending dict. ``AutoWeightsLoader``
invokes the model's ``load_weights`` once per contiguous run of the ``model.``
prefix (``itertools.groupby``), so an ``lm_head.weight`` between a scale and
its weight in the stream produces two calls. GLM-5.3's first shard stores
exactly that order (the F32 scales, then ``lm_head.weight``, then the FP8
weights), and a streaming loader that yields a shard's tensors in file order
(Run:ai) delivers it as is. The pair must still complete; with the pending
dict local to one call, the scale is dropped when the first call returns and
``wk_weights_proj`` is never written.

These tests run on CPU and drive the real ``load_weights`` implementations on
minimal module trees, in the style of ``tests/models/glm5next``.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v32.nvidia.model import (
    DeepseekV32ForCausalLM,
    DeepseekV32Model,
)

pytestmark = pytest.mark.cpu_test

HIDDEN = 256
BLOCK = 128
WK_ROWS = 128  # index_head_dim
WP_ROWS = 32  # index_n_heads (weights_proj, shard 1 of the fused param)
LAYER = 1
INDEXER = f"layers.{LAYER}.self_attn.indexer"
FUSED = f"{INDEXER}.wk_weights_proj.weight"
WEIGHT = f"{INDEXER}.wk.weight"
SCALE = f"{INDEXER}.wk.weight_scale_inv"


def _wk_pair() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    weight = (torch.randn(WK_ROWS, HIDDEN, generator=g) * 0.1).to(
        torch.float8_e4m3fn
    )
    scale = torch.full(
        (WK_ROWS // BLOCK, HIDDEN // BLOCK), 0.5, dtype=torch.float32
    )
    return {WEIGHT: weight, SCALE: scale}


def _make_model() -> tuple[DeepseekV32Model, list[int]]:
    """A ``DeepseekV32Model`` shell with only layer 1's fused indexer param.

    ``load_weights`` is the real implementation; the module tree carries just
    what it touches: ``config``, ``num_redundant_experts`` and a BF16
    ``wk_weights_proj.weight`` whose ``weight_loader`` writes shard 0 (wk)
    into the first rows and shard 1 (weights_proj) into the rest, recording
    the shards it received.
    """
    model = object.__new__(DeepseekV32Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        n_routed_experts=1, num_hidden_layers=LAYER + 1, num_nextn_predict_layers=0
    )
    model.num_redundant_experts = 0
    layers = nn.ModuleList([nn.Module() for _ in range(LAYER + 1)])
    attn, indexer, fused = nn.Module(), nn.Module(), nn.Module()
    fused.weight = nn.Parameter(
        torch.zeros(WK_ROWS + WP_ROWS, HIDDEN, dtype=torch.bfloat16),
        requires_grad=False,
    )
    shards_seen: list[int] = []

    def weight_loader(param: torch.Tensor, loaded: torch.Tensor, shard_id: int):
        shards_seen.append(shard_id)
        rows = slice(0, WK_ROWS) if shard_id == 0 else slice(WK_ROWS, None)
        param.data[rows].copy_(loaded)

    fused.weight.weight_loader = weight_loader
    indexer.wk_weights_proj = fused
    attn.indexer = indexer
    layers[LAYER].self_attn = attn
    model.layers = layers
    return model, shards_seen


def _assert_wk_written(model: DeepseekV32Model, shards_seen: list[int]) -> None:
    param = dict(model.named_parameters())[FUSED]
    pair = _wk_pair()
    expected = (pair[WEIGHT].float() * 0.5).to(torch.bfloat16)
    assert shards_seen == [0], shards_seen
    torch.testing.assert_close(param[:WK_ROWS], expected, rtol=0, atol=0)
    assert param[WK_ROWS:].abs().sum() == 0  # weights_proj is not in the stream


def test_wk_pair_completes_within_one_call() -> None:
    """Control: the weight and its scale arrive in one call."""
    model, shards_seen = _make_model()

    loaded = model.load_weights(iter(_wk_pair().items()))

    assert FUSED in loaded
    _assert_wk_written(model, shards_seen)


def test_wk_pair_split_across_load_weights_calls() -> None:
    """The scale in one call, its weight in the next (GLM-5.3's file order)."""
    model, shards_seen = _make_model()
    pair = _wk_pair()

    loaded = set()
    loaded |= model.load_weights(iter([(SCALE, pair[SCALE])]))
    loaded |= model.load_weights(iter([(WEIGHT, pair[WEIGHT])]))

    assert FUSED in loaded, (
        "wk_weights_proj was never written: the scale parked in the first "
        "call was discarded when it returned"
    )
    _assert_wk_written(model, shards_seen)


def _make_causal_lm() -> tuple[DeepseekV32ForCausalLM, DeepseekV32Model, list[int]]:
    text_model, shards_seen = _make_model()
    model = object.__new__(DeepseekV32ForCausalLM)
    nn.Module.__init__(model)
    model.model = text_model
    model.lm_head = nn.Module()
    model.lm_head.weight = nn.Parameter(torch.zeros(4), requires_grad=False)
    return model, text_model, shards_seen


@pytest.mark.parametrize(
    "order",
    [
        pytest.param(["lm_head", SCALE, WEIGHT], id="sorted-by-name"),
        pytest.param([SCALE, "lm_head", WEIGHT], id="lm-head-between-scale-and-weight"),
    ],
)
def test_stream_order_does_not_change_loaded_weights(order) -> None:
    """End to end through ``DeepseekV32ForCausalLM.load_weights``.

    Same three tensors, two orders. In the second (GLM-5.3's first shard, in
    file order), ``lm_head.weight`` splits the ``model.`` run in two, so
    ``AutoWeightsLoader`` calls ``DeepseekV32Model.load_weights`` twice. The
    result must not depend on it.
    """
    model, text_model, shards_seen = _make_causal_lm()
    pair = _wk_pair()
    lm_head = torch.tensor([1.0, 2.0, 3.0, 4.0])
    stream = [
        ("lm_head.weight", lm_head) if key == "lm_head" else (f"model.{key}", pair[key])
        for key in order
    ]

    loaded = model.load_weights(iter(stream))

    assert "lm_head.weight" in loaded
    assert model.lm_head.weight.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert f"model.{FUSED}" in loaded, (
        "wk_weights_proj was never written for this stream order"
    )
    _assert_wk_written(text_model, shards_seen)
