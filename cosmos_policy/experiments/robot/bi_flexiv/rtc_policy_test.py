from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_policy.experiments.robot.bi_flexiv.bi_flexiv_policy import (
    DreamTacBiFlexivPolicy, DreamTacBiFlexivPolicyConfig, PREPROCESSED_CAMERA_KEYS,
)
from cosmos_policy.models.policy_text2world_model import replace_latent_with_action_chunk


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter('dummy', torch.nn.Parameter(torch.zeros(1)))
        self.config = SimpleNamespace(state_t=11, min_num_conditional_frames=7, max_num_conditional_frames=7)
        self.tokenizer = SimpleNamespace(get_pixel_num_frames=lambda _: 41)
        self.batch = None

    def generate_samples_from_batch(self, batch, **kwargs):
        self.batch = batch
        actions = torch.zeros(1, 40, 20)
        if 'rtc_actions' in batch:
            actions[:, :batch['rtc_prefix_length']] = batch['rtc_actions'][:, :batch['rtc_prefix_length']]
        return replace_latent_with_action_chunk(torch.zeros(1, 16, 11, 28, 28), actions, torch.tensor([7]))


def make_policy(mode='q99', output='absolute_from_state'):
    stats = {'action_chunk_size': 40, 'gripper_start_idx': 18}
    for key in ('actions', 'proprio'):
        for suffix in ('q01', 'min'):
            stats[f'{key}_{suffix}'] = -np.ones(20)
        for suffix in ('q99', 'max'):
            stats[f'{key}_{suffix}'] = np.ones(20)
    policy = DreamTacBiFlexivPolicy(
        DreamTacBiFlexivPolicyConfig('', '', '', 'task', diffusion_step_cache=False,
                                    normalization_mode=mode, action_output=output),
        model=Model(), dataset_stats=stats, text_embeddings={'task': torch.zeros(2, 4)},
    )
    state = np.zeros(20, dtype=np.float32)
    state[[3, 7, 12, 16]] = 1
    state[[0, 9]] = 0.25
    obs = dict(state=state, images={k: np.zeros((4, 224, 224, 3), dtype=np.uint8)
                                  for k in PREPROCESSED_CAMERA_KEYS},
               tactile_self_attn_gate=np.zeros(2, dtype=np.float32))
    return policy, obs


@pytest.mark.parametrize('mode', ['q99', 'min_max'])
def test_prefix_is_rebased_without_clipping_and_preserved_exactly(mode):
    policy, obs = make_policy(mode)
    previous = np.tile(obs['state'], (20, 1))
    previous[:, 0] = 3.25  # Outside normalization bounds: do not clip old targets.
    previous[:, 18:] = [0.31, 0.72]
    original = previous.copy()
    result = policy.infer(obs, prev_chunk_left_over=previous, inference_delay=17, execution_horizon=40)
    np.testing.assert_array_equal(result['actions'][:17], previous[:17])
    np.testing.assert_array_equal(previous, original)
    normalized = policy.model.batch['rtc_actions'].numpy()[0]
    np.testing.assert_allclose(normalized[:17, 0], 3)
    np.testing.assert_allclose(normalized[:17, 18:], previous[:17, 18:], atol=1e-6)
    np.testing.assert_allclose(result['observation_relative_actions'][:17, 0], 3)
    assert result['rtc_prefix_length'] == 17
    # Later request has a different observation reference frame.
    obs['state'][0] += 0.5
    result = policy.infer(obs, prev_chunk_left_over=previous, inference_delay=17)
    np.testing.assert_array_equal(result['actions'][:17], previous[:17])
    np.testing.assert_allclose(policy.model.batch['rtc_actions'][0, :17, 0], 2.5)


def test_startup_zero_prefix_and_next_request_do_not_retain_constraints():
    policy, obs = make_policy()
    baseline = policy.infer(obs)['actions']
    startup = policy.infer(obs, prev_chunk_left_over=None, inference_delay=17)
    np.testing.assert_array_equal(baseline, startup['actions'])
    zero = policy.infer(obs, prev_chunk_left_over=baseline, inference_delay=0)
    np.testing.assert_array_equal(baseline, zero['actions'])
    policy.infer(obs, prev_chunk_left_over=baseline, inference_delay=17)
    again = policy.infer(obs)
    assert 'rtc_actions' not in policy.model.batch
    np.testing.assert_array_equal(baseline, again['actions'])


def test_reject_relative_queue_and_unknown_arguments():
    policy, obs = make_policy(output='observation_relative')
    assert policy.metadata['rtc_supported'] is False
    with pytest.raises(ValueError, match='absolute_from_state'):
        policy.infer(obs, prev_chunk_left_over=np.zeros((17, 20)), inference_delay=17)
    with pytest.raises(TypeError, match='Unknown'):
        policy.infer(obs, typo_delay=17)
    with pytest.raises(ValueError, match='execution_horizon'):
        policy.infer(obs, execution_horizon=41)


def test_websocket_rtc_contract_and_exact_prefix():
    from cosmos_policy.experiments.robot.bi_flexiv.rtc_benchmark import loopback_client
    policy, obs = make_policy()
    with loopback_client(policy) as infer:
        baseline = infer(obs)
        prefix = baseline['actions'][-17:].copy()
        obs['state'][0] += 0.1
        result = infer(obs, prev_chunk_left_over=prefix, inference_delay=17, execution_horizon=40)
        np.testing.assert_array_equal(result['actions'][:17], prefix)
        assert result['rtc_prefix_length'] == 17
        assert result['server_timing']['infer_ms'] > 0
