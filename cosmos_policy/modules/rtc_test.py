from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_policy.modules.rtc import ActionPrefixInpainting, prefix_from_latency, validate_prefix
from cosmos_policy.modules.cosmos_sampler import CosmosPolicySampler
from cosmos_policy.models.policy_text2world_model import CosmosPolicyDiffusionModel
from cosmos_policy.experiments.robot.cosmos_utils import extract_action_chunk_from_latent_sequence


@pytest.mark.parametrize('delay', [-1, 40, 1.5, True])
def test_reject_invalid_delay(delay):
    with pytest.raises(ValueError):
        validate_prefix(np.zeros((40, 20)), delay, horizon=40, action_dim=20)


def test_startup_and_short_prefix():
    assert validate_prefix(None, 17, horizon=40, action_dim=20) == (None, 0)
    for previous in [np.zeros((2, 20)), np.full((20, 20), np.nan), np.zeros((20, 19))]:
        with pytest.raises(ValueError):
            validate_prefix(previous, 17, horizon=40, action_dim=20)
    assert prefix_from_latency(0.469) == 17
    # Do not clamp an infeasible delay into an apparently feasible answer.
    assert prefix_from_latency(1.4) == 44


@pytest.mark.parametrize('steps', [1, 5, 10])
def test_production_sampler_inpaints_every_repeat_and_conditions_suffix(steps):
    initial = torch.randn(1, 16, 11, 28, 28) * 80
    actions = torch.randn(1, 40, 20)
    indices = torch.tensor([7])
    constraint = ActionPrefixInpainting(initial, actions, indices, 17, 80)
    count = 0

    def denoise(value, sigma):
        nonlocal count
        count += 1
        expected = constraint.clean + sigma.reshape(1, 1, 1, 1, 1) * constraint.noise
        torch.testing.assert_close(value[constraint.mask], expected[constraint.mask])
        # The suffix depends on the actual prefix input, not an output splice.
        return torch.ones_like(value) * value[constraint.mask].mean()

    result = CosmosPolicySampler()(constraint.wrap(denoise), initial, num_steps=steps)
    extracted = extract_action_chunk_from_latent_sequence(result, (40, 20), indices)
    torch.testing.assert_close(extracted[:, :17], actions[:, :17])
    assert count == steps
    # Partial repeat occupies the tail of the action latent too.
    expected_mask = (torch.arange(16 * 28 * 28) % 800 < 17 * 20).reshape(16, 28, 28)
    assert torch.equal(constraint.mask[0, :, 7], expected_mask)
    assert not constraint.mask[:, :, :7].any()
    assert not constraint.mask[:, :, 8:].any()


def test_generate_samples_routes_rtc_into_every_denoising_call():
    initial = torch.randn(1, 16, 11, 28, 28) * 80
    actions = torch.randn(1, 40, 20)
    seen = []
    def get_x0(*args, **kwargs):
        def denoise(value, sigma):
            seen.append(value.clone())
            return value * 0.1
        return denoise
    harness = SimpleNamespace(
        _normalize_video_databatch_inplace=lambda batch: None,
        _augment_image_dim_inplace=lambda batch: None,
        is_image_batch=lambda batch: False,
        input_data_key='video',
        get_x0_fn_from_batch=get_x0,
        config=SimpleNamespace(use_flowunipc_scheduler=False),
        net=SimpleNamespace(is_context_parallel_enabled=False),
        sde=SimpleNamespace(sigma_max=80, sigma_min=0.002),
        sampler=CosmosPolicySampler(),
    )
    batch = {'video': torch.zeros(1), 'rtc_actions': actions,
             'rtc_prefix_length': 17, 'action_latent_idx': torch.tensor([7])}
    result = CosmosPolicyDiffusionModel.generate_samples_from_batch(
        harness, batch, n_sample=1, state_shape=initial.shape[1:], x_sigma_max=initial, num_steps=5)
    extracted = extract_action_chunk_from_latent_sequence(result, (40, 20), batch['action_latent_idx'])
    torch.testing.assert_close(extracted[:, :17], actions[:, :17])
    assert len(seen) == 5
    constraint = ActionPrefixInpainting(initial, actions, batch['action_latent_idx'], 17, 80)
    torch.testing.assert_close(seen[0][constraint.mask], (constraint.clean + initial)[constraint.mask])
