from __future__ import annotations

from types import SimpleNamespace

import torch

from cosmos_policy.experiments.robot.cosmos_utils import extract_action_chunk_from_latent_sequence
from cosmos_policy.models.policy_text2world_model import (
    CosmosPolicyDiffusionModel,
    replace_latent_with_action_chunk,
)


def _policy_batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
    index = lambda value: torch.full((batch_size,), value, dtype=torch.long)
    zeros = torch.zeros(batch_size, dtype=torch.long)
    return {
        "actions": torch.zeros(batch_size, 30, 20),
        "proprio": torch.zeros(batch_size, 20),
        "action_latent_idx": index(7),
        "current_proprio_latent_idx": index(1),
        "future_proprio_latent_idx": index(-1),
        "future_wrist_image_latent_idx": index(9),
        "future_wrist_image2_latent_idx": index(10),
        "future_image_latent_idx": index(8),
        "rollout_data_mask": zeros.clone(),
        "world_model_sample_mask": zeros.clone(),
        "value_function_sample_mask": zeros.clone(),
        "value_function_return": torch.full((batch_size,), -100.0),
        "value_latent_idx": index(-1),
    }


def test_action_latent_injection_and_extraction_are_exact_inverses() -> None:
    action = torch.linspace(-1.0, 1.0, 24).reshape(2, 3, 4)
    action_indices = torch.tensor([2, 7])
    latent = torch.zeros(2, 3, 11, 4, 5)

    injected = replace_latent_with_action_chunk(latent, action, action_indices)
    extracted = extract_action_chunk_from_latent_sequence(injected, (3, 4), action_indices)

    torch.testing.assert_close(extracted, action)


def test_30_step_bi_flexiv_action_chunk_fits_the_real_latent_shape() -> None:
    action = torch.linspace(-1.0, 1.0, 2 * 30 * 20).reshape(2, 30, 20)
    action_indices = torch.tensor([7, 7])
    latent = torch.zeros(2, 16, 11, 28, 28)

    injected = replace_latent_with_action_chunk(latent, action, action_indices)
    extracted = extract_action_chunk_from_latent_sequence(injected, (30, 20), action_indices)

    assert extracted.shape == (2, 30, 20)
    torch.testing.assert_close(extracted, action)


def test_training_step_accepts_11_slot_batch_without_future_proprio() -> None:
    captured: dict[str, object] = {}
    x0 = torch.zeros(1, 1, 11, 1, 1)
    harness = SimpleNamespace(
        config=SimpleNamespace(text_encoder_config=None),
        loss_reduce="mean",
        loss_scale=1.0,
        _update_train_stats=lambda _batch: None,
        get_data_and_condition=lambda _batch: (None, x0, object()),
        draw_training_sigma_and_epsilon=lambda _size, _condition: (
            torch.ones(1, 1),
            torch.zeros_like(x0),
        ),
        broadcast_split_for_model_parallelsim=lambda x, condition, epsilon, sigma: (
            x,
            condition,
            epsilon,
            sigma,
        ),
    )

    def _compute(*_args: object, **kwargs: object):
        captured.update(kwargs)
        elementwise_loss = torch.ones_like(x0)
        return {}, elementwise_loss, elementwise_loss, elementwise_loss

    harness.compute_loss_with_epsilon_and_sigma = _compute
    _, loss = CosmosPolicyDiffusionModel.training_step(harness, _policy_batch(), iteration=0)

    assert captured["future_proprio"] is None
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_11_slot_loss_without_future_proprio_has_finite_backward() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.25))
    config = SimpleNamespace(
        mask_current_state_action_for_value_prediction=False,
        mask_future_state_for_qvalue_prediction=False,
        mask_loss_for_action_future_state_prediction=False,
        mask_value_prediction_loss_for_policy_prediction=False,
        action_loss_multiplier=1,
    )

    class _SDE:
        @staticmethod
        def marginal_prob(x0: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x0, sigma

    harness = SimpleNamespace(
        config=config,
        sde=_SDE(),
        denoise=lambda xt, _sigma, _condition: SimpleNamespace(x0=parameter.expand_as(xt)),
        get_per_sigma_loss_weights=lambda sigma: torch.ones_like(sigma),
    )
    batch = _policy_batch()
    x0 = torch.zeros(1, 2, 11, 20, 20)
    condition = SimpleNamespace()
    output, elementwise_loss, _, _ = CosmosPolicyDiffusionModel.compute_loss_with_epsilon_and_sigma(
        harness,
        x0,
        condition,
        torch.zeros_like(x0),
        torch.ones(1, 1),
        action_chunk=batch["actions"],
        action_indices=batch["action_latent_idx"],
        proprio=batch["proprio"],
        current_proprio_indices=batch["current_proprio_latent_idx"],
        future_proprio=None,
        future_proprio_indices=batch["future_proprio_latent_idx"],
        future_wrist_image_indices=batch["future_wrist_image_latent_idx"],
        future_wrist_image2_indices=batch["future_wrist_image2_latent_idx"],
        future_image_indices=batch["future_image_latent_idx"],
        future_image2_indices=None,
        future_tactile_left_indices=None,
        future_tactile_right_indices=None,
        future_tactile_indices=None,
        rollout_data_mask=batch["rollout_data_mask"],
        world_model_sample_mask=batch["world_model_sample_mask"],
        value_function_sample_mask=batch["value_function_sample_mask"],
        value_function_return=batch["value_function_return"],
        value_indices=batch["value_latent_idx"],
    )
    loss = elementwise_loss.mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad)
    assert output["x0"].shape == (1, 2, 11, 20, 20)
