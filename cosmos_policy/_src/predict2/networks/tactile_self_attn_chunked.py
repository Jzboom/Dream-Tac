# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Chunked self-attention with outer-product logits bias: scores_ij += gamma_b * a_i * b_j.

## Can we use Dao-AILab FlashAttention (flash_attn_func / FA3) here?

**Not with the upstream public API today.**

- `flash_attn.flash_attn_interface.flash_attn_func` only supports optional **ALiBi**-style bias:
  ``(-alibi_slope * |i - j|)`` per head, not a general per-(i,j) matrix and **not** ``gamma * a_i * b_j``
  with arbitrary vectors ``a``, ``b`` on latent token positions.
- Hopper **FA3** (`flash_attn_3`) exposes the same ALiBi-style mechanism, not arbitrary additive logits.
- Community PRs (e.g. custom dense bias) are not a stable, drop-in API for this rank-1 outer product.

So **you cannot** call `flash_attn_func(q, k, v, ...)` and pass our tactile bias without either:
- materializing a full (or chunked) bias tile and using an op that accepts it (what we do with SDPA), or
- a **custom CUDA/Triton** kernel (or projects like FlashBias / future PyTorch AttnBias types) that fuses
  softmax with this bias structure.

## FlashBias-style SDPA (rank-1 outer bias) — recommended for speed

Our bias is **rank-1**: ``gamma_b * a_i * b_j = (sqrt(gamma_b)*a_i) * (sqrt(gamma_b)*b_j)``.

Following [FlashBias](https://arxiv.org/pdf/2505.12044) (NeurIPS 2025, Tsinghua), this is a self-contained
re-implementation of the trick (no external FlashBias dependency required):
use **concatenated Q/K** so the extra inner-product dimension contributes exactly that outer product, with
**no float attn_mask** — so PyTorch SDPA can select **Flash / cuDNN FMHA** backends (subject to head
dim rules: ``(headdim + rank)`` padded to a multiple of 8).

Backend ``flashbias_sdpa``: one fused SDPA over the **full** sequence, or two
rectangular calls for block-causal attention (``chunk_q`` ignored).

## Other backends

- **flashbias_sdpa** (default): full-sequence SDPA with **concat** ``[q*scale, q_bias], [k, k_bias]`` (FlashBias trick); no float ``attn_mask`` — best chance to match **pre-bias FlashAttention speed**.
- **sdpa**: per query-chunk SDPA with float ``attn_mask`` (Flash often disabled).
- **eager**: explicit ``matmul -> +bias -> softmax -> @V`` (debug).

Env: ``COSMOS_TACTILE_SELF_ATTN_BACKEND`` = ``flashbias_sdpa`` | ``sdpa`` | ``eager``.
Optional: ``COSMOS_TACTILE_SELF_ATTN_SDP_PRIORITY`` (see below).
"""

from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _HAS_SDPA_KERNEL = True
except Exception:  # pragma: no cover
    SDPBackend = None  # type: ignore[misc, assignment]
    sdpa_kernel = None  # type: ignore[misc, assignment]
    _HAS_SDPA_KERNEL = False

TactileChunkBackend = Literal["sdpa", "eager", "flashbias_sdpa"]


def _sdp_priority_from_env() -> list | None:
    """Map COSMOS_TACTILE_SELF_ATTN_SDP_PRIORITY to backend order; None = PyTorch default."""
    if not _HAS_SDPA_KERNEL:
        return None
    key = os.environ.get("COSMOS_TACTILE_SELF_ATTN_SDP_PRIORITY", "").strip().lower()
    if not key:
        return None
    order = []
    for part in key.replace(",", " ").split():
        part = part.strip()
        if part in ("flash", "flash_attention"):
            order.append(SDPBackend.FLASH_ATTENTION)
        elif part in ("cudnn",):
            order.append(SDPBackend.CUDNN_ATTENTION)
        elif part in ("efficient", "mem_efficient", "memory_efficient"):
            order.append(SDPBackend.EFFICIENT_ATTENTION)
        elif part in ("math",):
            order.append(SDPBackend.MATH)
    return order or None


def _scaled_dot_product_attention_chunk(
    q_c: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    backends = _sdp_priority_from_env()
    if backends is not None:
        try:
            with sdpa_kernel(backends=backends, set_priority_order=True):
                return F.scaled_dot_product_attention(
                    q_c,
                    k,
                    v,
                    attn_mask=attn_bias,
                    is_causal=False,
                    scale=scale,
                )
        except Exception:
            pass
    return F.scaled_dot_product_attention(
        q_c,
        k,
        v,
        attn_mask=attn_bias,
        is_causal=False,
        scale=scale,
    )


def _chunk_eager(
    q_c: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    scores = torch.matmul(q_c, k.transpose(-2, -1)) * scale
    scores = scores + attn_bias
    attn_w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=q_c.dtype)
    return torch.matmul(attn_w, v)


def _flashbias_sdpa_full(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    q_bias_bsh1: torch.Tensor,
    k_bias_bsh1: torch.Tensor,
    softmax_scale: float,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """
    FlashBias SDPA formulation (concat extra dim, scale=1): see FlashBias README / attention_func.flashbias_sdpa.

    q/q_bias: (B, Sq, H, D/1); k/v/k_bias: (B, Sk, H, D/D/1).
    Internally uses (B, H, S, *) for F.scaled_dot_product_attention.
    """
    q = q_bshd.transpose(1, 2)
    k = k_bshd.transpose(1, 2)
    v = v_bshd.transpose(1, 2)
    qb = q_bias_bsh1.transpose(1, 2)
    kb = k_bias_bsh1.transpose(1, 2)
    _, _, _, d_h = q.shape
    _, _, _, r_b = qb.shape
    total = d_h + r_b
    pad = (8 - (total % 8)) % 8

    # PyTorch's CUDA FlashAttention requires Q, K, and V to have the same
    # head dimension.  The rank-1 FlashBias feature makes Q/K wider than V;
    # append zero value channels as well and discard their identically-zero
    # outputs afterwards.  Without this, SDPA rejects the flash kernel and
    # silently falls back to a much slower backend under the default policy.
    attention_dim = total + pad
    value_pad = attention_dim - v.shape[-1]
    if value_pad < 0:
        raise ValueError(
            f"FlashBias attention dimension {attention_dim} is smaller than value dimension {v.shape[-1]}"
        )
    if value_pad:
        v_attention = F.pad(v, (0, value_pad))
    else:
        v_attention = v

    def _run(q_cat: torch.Tensor, k_cat: torch.Tensor) -> torch.Tensor:
        backends = _sdp_priority_from_env()
        if backends is not None:
            try:
                with sdpa_kernel(backends=backends, set_priority_order=True):
                    return F.scaled_dot_product_attention(
                        q_cat,
                        k_cat,
                        v_attention,
                        attn_mask=None,
                        dropout_p=0.0,
                        scale=1.0,
                        is_causal=causal,
                    )
            except Exception:
                pass
        return F.scaled_dot_product_attention(
            q_cat,
            k_cat,
            v_attention,
            attn_mask=None,
            dropout_p=0.0,
            scale=1.0,
            is_causal=causal,
        )

    if pad == 0:
        out = _run(torch.cat([q * softmax_scale, qb], dim=-1), torch.cat([k, kb], dim=-1))
    else:
        q_blank = torch.zeros(q.shape[0], q.shape[1], q.shape[2], pad, device=q.device, dtype=q.dtype)
        k_blank = torch.zeros(k.shape[0], k.shape[1], k.shape[2], pad, device=k.device, dtype=k.dtype)
        out = _run(
            torch.cat([q * softmax_scale, qb, q_blank], dim=-1),
            torch.cat([k, kb, k_blank], dim=-1),
        )
    out = out[..., : v.shape[-1]]
    return out.transpose(1, 2).contiguous()


def dao_flash_attn_supports_tactile_outer_bias() -> bool:
    """
    Returns True only if we detect a *future* flash_attn API that accepts a general logits bias
    compatible with our chunked (B, H, Lq, Lk) bias. Upstream flash_attn_func / FA3: False.
    """
    try:
        import inspect

        from flash_attn.flash_attn_interface import flash_attn_func
    except Exception:
        try:
            import inspect

            from flash_attn_3.flash_attn_interface import flash_attn_func
        except Exception:
            return False
    try:
        params = inspect.signature(flash_attn_func).parameters
    except Exception:
        return False
    for name in ("attn_bias", "attention_bias", "bias", "attn_logits_bias"):
        if name in params:
            return True
    return False


def block_causal_attention(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    condition_prefix_tokens: int,
) -> torch.Tensor:
    """Two-block self-attention without materializing a dense mask.

    Condition queries attend bidirectionally only within the condition prefix;
    prediction queries attend bidirectionally to the complete sequence.
    """
    sequence_length = q_B_S_H_D.shape[1]
    if k_B_S_H_D.shape[1] != sequence_length or v_B_S_H_D.shape[1] != sequence_length:
        raise ValueError("Block-causal self-attention requires equal full Q/K/V sequence lengths")
    if not 0 < condition_prefix_tokens < sequence_length:
        raise ValueError(
            f"condition_prefix_tokens must be in (0, {sequence_length}), got {condition_prefix_tokens}"
        )

    def _attend(q_part: torch.Tensor, k_part: torch.Tensor, v_part: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q_part.transpose(1, 2),
            k_part.transpose(1, 2),
            v_part.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)

    condition = _attend(
        q_B_S_H_D[:, :condition_prefix_tokens],
        k_B_S_H_D[:, :condition_prefix_tokens],
        v_B_S_H_D[:, :condition_prefix_tokens],
    )
    prediction = _attend(q_B_S_H_D[:, condition_prefix_tokens:], k_B_S_H_D, v_B_S_H_D)
    return torch.cat((condition, prediction), dim=1).contiguous()


def self_attention_with_tactile_outer_bias_chunked(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    a_S: torch.Tensor,
    b_S: torch.Tensor,
    gamma_B: torch.Tensor,
    chunk_q: int,
    output_proj: nn.Linear,
    output_dropout: nn.Module,
    gamma_BS: torch.Tensor | None = None,
    condition_prefix_tokens: int = 0,
) -> torch.Tensor:
    """
    Self-attention with additive logits bias: scores_ij += gamma_b * a_i * b_j (batched gamma).
    q,k,v: (B, S, H, D).

    - ``flashbias_sdpa``: one SDPA call (two when block causal), no float mask;
      ``chunk_q`` ignored.
    - ``sdpa`` / ``eager``: query-axis chunking with ``chunk_q``.
    """
    # Default: FlashBias-style concat (no float attn_mask) so SDPA can use Flash/cuDNN FMHA when supported.
    backend: str = os.environ.get("COSMOS_TACTILE_SELF_ATTN_BACKEND", "flashbias_sdpa").strip().lower()
    if backend not in ("sdpa", "eager", "flashbias_sdpa"):
        backend = "sdpa"

    B, S, Hn, D = q_B_S_H_D.shape
    dtype = q_B_S_H_D.dtype
    device = q_B_S_H_D.device
    a = a_S.to(device=device, dtype=dtype)
    b = b_S.to(device=device, dtype=dtype)
    if gamma_BS is None:
        sqrt_gamma = torch.sqrt(gamma_B.to(device=device, dtype=dtype).clamp(min=0.0)).view(B, 1, 1, 1)
        key_bias_scale = sqrt_gamma * b.view(1, S, 1, 1)
    else:
        sqrt_gamma = torch.ones(B, 1, 1, 1, device=device, dtype=dtype)
        key_bias_scale = gamma_BS.to(device=device, dtype=dtype).clamp(min=0.0).view(B, S, 1, 1)
    q_bias = sqrt_gamma * a.view(1, S, 1, 1).expand(B, S, Hn, 1)
    k_bias = key_bias_scale.expand(B, S, Hn, 1)

    def _attend(
        q_part: torch.Tensor,
        k_part: torch.Tensor,
        v_part: torch.Tensor,
        q_bias_part: torch.Tensor,
        k_bias_part: torch.Tensor,
    ) -> torch.Tensor:
        if backend == "flashbias_sdpa":
            return _flashbias_sdpa_full(
                q_part,
                k_part,
                v_part,
                q_bias_part,
                k_bias_part,
                softmax_scale=D**-0.5,
                causal=False,
            )

        q = rearrange(q_part, "b s h d -> b h s d")
        k = rearrange(k_part, "b s h d -> b h s d")
        v = rearrange(v_part, "b s h d -> b h s d")
        qb = rearrange(q_bias_part, "b s h r -> b h s r")
        kb = rearrange(k_bias_part, "b s h r -> b h r s")
        out = torch.empty_like(q)
        for qs in range(0, q.shape[2], chunk_q):
            qe = min(qs + chunk_q, q.shape[2])
            q_c = q[:, :, qs:qe, :]
            attn_bias = qb[:, :, qs:qe, :] * kb
            if backend == "eager":
                out[:, :, qs:qe, :] = _chunk_eager(q_c, k, v, attn_bias, D**-0.5)
            else:
                out[:, :, qs:qe, :] = _scaled_dot_product_attention_chunk(q_c, k, v, attn_bias, D**-0.5)
        return rearrange(out, "b h s d -> b s h d")

    if condition_prefix_tokens:
        if not 0 < condition_prefix_tokens < S:
            raise ValueError(f"condition_prefix_tokens must be in (0, {S}), got {condition_prefix_tokens}")
        condition = _attend(
            q_B_S_H_D[:, :condition_prefix_tokens],
            k_B_S_H_D[:, :condition_prefix_tokens],
            v_B_S_H_D[:, :condition_prefix_tokens],
            q_bias[:, :condition_prefix_tokens],
            k_bias[:, :condition_prefix_tokens],
        )
        prediction = _attend(
            q_B_S_H_D[:, condition_prefix_tokens:],
            k_B_S_H_D,
            v_B_S_H_D,
            q_bias[:, condition_prefix_tokens:],
            k_bias,
        )
        out_B_S_H_D = torch.cat((condition, prediction), dim=1)
    else:
        out_B_S_H_D = _attend(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D, q_bias, k_bias)
    flat = rearrange(out_B_S_H_D, "b s h d -> b s (h d)")
    return output_dropout(output_proj(flat))
