from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
from torch import nn

from cosmos_policy._src.predict2.networks.minimal_v4_dit import MiniTrainDIT
from cosmos_policy._src.predict2.networks.tactile_self_attn_chunked import (
    _flashbias_sdpa_full,
    self_attention_with_tactile_outer_bias_chunked,
)


def _run_attention(
    backend: str,
    monkeypatch,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    projection: nn.Linear,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    monkeypatch.setenv("COSMOS_TACTILE_SELF_ATTN_BACKEND", backend)
    q, k, v = (tensor.detach().clone().requires_grad_(True) for tensor in inputs)
    projection = copy.deepcopy(projection)
    a = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=q.dtype)
    b = 1.0 - a
    gamma_bs = torch.tensor(
        [[0.0, 0.0, 0.4, 1.2, 0.0, 0.0], [0.0, 0.0, 1.4, 0.2, 0.0, 0.0]],
        dtype=q.dtype,
    )
    output = self_attention_with_tactile_outer_bias_chunked(
        q,
        k,
        v,
        a,
        b,
        gamma_B=torch.ones(q.shape[0], dtype=q.dtype),
        gamma_BS=gamma_bs,
        chunk_q=2,
        output_proj=projection,
        output_dropout=nn.Identity(),
    )
    gradients = torch.autograd.grad(output.square().sum(), (q, k, v, *projection.parameters()))
    return output.detach(), tuple(gradient.detach() for gradient in gradients)


def test_flashbias_matches_eager_forward_and_backward(monkeypatch) -> None:
    generator = torch.Generator().manual_seed(903)
    inputs = tuple(torch.randn(2, 6, 2, 4, generator=generator, dtype=torch.float64) for _ in range(3))
    projection = nn.Linear(8, 8, dtype=torch.float64)

    eager_output, eager_gradients = _run_attention("eager", monkeypatch, inputs, projection)
    flash_output, flash_gradients = _run_attention("flashbias_sdpa", monkeypatch, inputs, projection)

    # CPU SDPA may internally use float32 accumulation even for float64 input.
    torch.testing.assert_close(flash_output, eager_output, rtol=1e-6, atol=1e-7)
    for flash_gradient, eager_gradient in zip(flash_gradients, eager_gradients):
        torch.testing.assert_close(flash_gradient, eager_gradient, rtol=1e-5, atol=2e-7)


def test_flashbias_pads_value_to_the_same_dimension_as_query_and_key(monkeypatch) -> None:
    seen_dimensions: list[tuple[int, int, int]] = []

    def _fake_sdpa(q, k, v, **_kwargs):
        seen_dimensions.append((q.shape[-1], k.shape[-1], v.shape[-1]))
        return v

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", _fake_sdpa)
    q = torch.randn(1, 3, 2, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    q_bias = torch.ones(1, 3, 2, 1)
    k_bias = torch.ones_like(q_bias)

    output = _flashbias_sdpa_full(q, k, v, q_bias, k_bias, softmax_scale=0.5)

    assert seen_dimensions == [(8, 8, 8)]
    assert output.shape == v.shape
    torch.testing.assert_close(output, v)


def test_grouped_gate_maps_only_to_the_two_merged_tactile_slots() -> None:
    harness = SimpleNamespace(
        use_tactile_self_attn_bias=True,
        tactile_self_attn_alpha=2.0,
        tactile_latent_t_indices=(5, 6),
        tactile_latent_gate_groups=(0, 1),
        tactile_attn_chunk_q=32,
    )
    x = torch.zeros(2, 11, 2, 3, 4)
    gate = torch.tensor([[0.2, 0.7], [0.4, 0.9]])

    attention_kwargs = MiniTrainDIT._tactile_self_attn_block_kw(harness, x, gate)
    gamma_bs = attention_kwargs["tactile_gamma_BS"].reshape(2, 11, 6)
    expected = torch.zeros_like(gamma_bs)
    expected[:, 5] = 2.0 * gate[:, 0, None]
    expected[:, 6] = 2.0 * gate[:, 1, None]

    torch.testing.assert_close(gamma_bs, expected)
    torch.testing.assert_close(
        attention_kwargs["tactile_outer_b_S"].reshape(11, 6).sum(dim=1),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0]),
    )
