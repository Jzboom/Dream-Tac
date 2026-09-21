"""Inference-only RTC inpainting for the repeated action latent layout.

Uses the Xense RTC wire contract, but conditions EDM samples at their current
noise level. Unlike pi0's training-time RTC, this needs no per-action time
embedding or retraining. Only the carried prefix is constrained.
"""

import math

import numpy as np
import torch


def prefix_from_latency(latency_seconds: float, frequency_hz: float = 30.0, margin: int = 2) -> int:
    if not math.isfinite(latency_seconds) or latency_seconds < 0:
        raise ValueError("latency_seconds must be finite and nonnegative")
    if not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be finite and positive")
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
        raise ValueError("margin must be a nonnegative integer")
    return math.ceil(latency_seconds * frequency_hz) + margin


def validate_prefix(prev_chunk_left_over, inference_delay, *, horizon: int, action_dim: int):
    """Never silently freeze padding when the client has too few actions left."""
    delay = 0 if inference_delay is None else inference_delay
    if isinstance(delay, (bool, np.bool_)) or not isinstance(delay, (int, np.integer)):
        raise ValueError("inference_delay must be an integer action count")
    if not 0 <= delay < horizon:
        raise ValueError(f"inference_delay must be in [0, {horizon - 1}]")
    if prev_chunk_left_over is None:
        # Xense warmup sends None with a default nonzero delay.
        return None, 0
    previous = np.array(prev_chunk_left_over, dtype=np.float32, copy=True)
    if previous.ndim != 2 or previous.shape[1] != action_dim or len(previous) > horizon:
        raise ValueError(f"prev_chunk_left_over must have shape (remaining <= {horizon}, {action_dim})")
    if len(previous) < delay or not np.isfinite(previous).all():
        raise ValueError("RTC prefix is shorter than inference_delay or contains NaN/Inf")
    return previous[:delay].copy(), int(delay)


class ActionPrefixInpainting:
    """Constrain every repeated prefix coordinate, including the partial repeat."""

    def __init__(self, initial, actions, indices, prefix_length, sigma_max):
        if actions.ndim != 3 or actions.shape[0] != initial.shape[0]:
            raise ValueError("RTC actions must be (batch, horizon, action_dim)")
        if not 0 < prefix_length < actions.shape[1]:
            raise ValueError("RTC prefix must leave at least one new action")
        batch, channels, _, height, width = initial.shape
        elements = channels * height * width
        action_elements = actions.shape[1] * actions.shape[2]
        if elements < action_elements:
            raise ValueError("Action chunk does not fit latent frame")
        positions = torch.arange(elements, device=initial.device) % action_elements
        frame_mask = (positions < prefix_length * actions.shape[2]).reshape(1, channels, height, width)
        frame = actions.to(initial).reshape(batch, -1)[:, positions].reshape(batch, channels, height, width)
        self.mask = torch.zeros_like(initial, dtype=torch.bool)
        self.clean = torch.zeros_like(initial)
        batch_indices = torch.arange(batch, device=initial.device)
        self.mask[batch_indices, :, indices] = frame_mask
        self.clean[batch_indices, :, indices] = frame
        self.noise = initial / sigma_max

    def at_sigma(self, value, sigma):
        sigma = sigma.reshape(-1, *([1] * (value.ndim - 1)))
        return torch.where(self.mask, self.clean + sigma * self.noise, value)

    def finish(self, value):
        return torch.where(self.mask, self.clean, value)

    def wrap(self, denoise):
        def constrained(value, sigma):
            return self.finish(denoise(self.at_sigma(value, sigma), sigma))
        return constrained
