"""Legacy experiment-name aliases for the current 11-slot bi_flexiv policy."""

from hydra.core.config_store import ConfigStore

from cosmos_policy.config.experiment.lerobot_bi_flexiv_experiment_configs import (
    cosmos_predict2_2b_480p_lerobot_bi_flexiv_wam_11slot,
    cosmos_predict2_2b_480p_lerobot_bi_flexiv_wam_11slot__inference_only,
)

cosmos_predict2_2b_480p_lerobot_earbud_tactile = cosmos_predict2_2b_480p_lerobot_bi_flexiv_wam_11slot
cosmos_predict2_2b_480p_lerobot_earbud_tactile__inference_only = (
    cosmos_predict2_2b_480p_lerobot_bi_flexiv_wam_11slot__inference_only
)

_LEGACY_EXPERIMENTS = {
    "cosmos_predict2_2b_480p_lerobot_earbud_tactile": cosmos_predict2_2b_480p_lerobot_earbud_tactile,
    "cosmos_predict2_2b_480p_lerobot_earbud_tactile__inference_only": (
        cosmos_predict2_2b_480p_lerobot_earbud_tactile__inference_only
    ),
}

for _name, _config in _LEGACY_EXPERIMENTS.items():
    ConfigStore.instance().store(group="experiment", package="_global_", name=_name, node=_config)
