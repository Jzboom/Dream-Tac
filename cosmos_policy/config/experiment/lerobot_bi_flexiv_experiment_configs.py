# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.utils import log
from cosmos_policy.config.local_paths import DEFAULT_LEROBOT_DATA_DIR
from cosmos_policy.datasets.lerobot_bi_flexiv_dataset import LeRobotBiFlexivDataset
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel

_LEROBOT_BI_FLEXIV_DATA_DIR = os.environ.get(
    "LEROBOT_BI_FLEXIV_DATA_DIR",
    os.environ.get("LEROBOT_EARBUD_DATA_DIR", str(DEFAULT_LEROBOT_DATA_DIR)),
)

lerobot_bi_flexiv_dataset = L(LeRobotBiFlexivDataset)(
    data_dir="${lerobot_dataset_path}",
    t5_text_embeddings_path="${lerobot_dataset_path}/t5_embeddings.pkl",
    chunk_size=20,
    final_image_size=224,
    normalize_images=False,
    normalize_actions=True,
    normalize_proprio=True,
    normalization_mode="q99",
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    num_duplicates_per_image=4,
    return_value_function_returns=False,
    gamma=0.99,
    gripper_start_idx=18,
    max_open_videos=32,
)

cosmos_predict2_2b_480p_lerobot_bi_flexiv_tactile = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        # Override from the training command with, for example:
        # lerobot_dataset_path=../another_lerobot_dataset
        lerobot_dataset_path=_LEROBOT_BI_FLEXIV_DATA_DIR,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=18,
                min_num_conditional_frames=9,
                max_num_conditional_frames=9,
                tokenizer=dict(
                    chunk_duration=69,
                ),
                net=dict(
                    use_tactile_self_attn_bias=True,
                    tactile_self_attn_alpha=2.0,
                    tactile_latent_t_indices=(5, 6, 7, 8, 14, 15, 16, 17),
                    tactile_latent_gate_groups=(0, 0, 1, 1, 0, 0, 1, 1),
                    tactile_attn_chunk_q=32,
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=2,
            persistent_workers=False,
            pin_memory=True,
            dataset=lerobot_bi_flexiv_dataset,
            sampler=L(DistributedSampler)(
                dataset=lerobot_bi_flexiv_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=8,
            drop_last=True,
        ),
        checkpoint=dict(
            save_iter=10000,
        ),
        job=dict(
            project="cosmos_policy_lerobot_bi_flexiv",
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_lerobot_bi_flexiv_tactile",
            wandb_entity=None,
            wandb_id=None,
        ),
    )
)

cosmos_predict2_2b_480p_lerobot_bi_flexiv_tactile__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_lerobot_bi_flexiv_tactile",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=dict(
                    sigma_max=40,
                    sigma_min=8,
                ),
            ),
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_lerobot_bi_flexiv_tactile__inference_only",
        ),
    )
)


def _register_configs() -> None:
    cs = ConfigStore.instance()
    for item in [
        cosmos_predict2_2b_480p_lerobot_bi_flexiv_tactile,
        cosmos_predict2_2b_480p_lerobot_bi_flexiv_tactile__inference_only,
    ]:
        experiment_name = item["job"]["name"]
        log.info(f"Registering experiment: {experiment_name}")
        cs.store(group="experiment", package="_global_", name=experiment_name, node=item)


_register_configs()
