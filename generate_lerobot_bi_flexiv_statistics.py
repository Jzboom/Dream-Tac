from pathlib import Path

from cosmos_policy.datasets.lerobot_bi_flexiv_dataset import LeRobotBiFlexivDataset


def main() -> None:
    data_dir = Path("../pick_up_cube_0713").resolve()
    print(f"Dataset: {data_dir}")

    LeRobotBiFlexivDataset(
        data_dir=str(data_dir),
        t5_text_embeddings_path="",
        chunk_size=20,
        gripper_start_idx=18,
        normalization_mode="q99",
        normalize_actions=True,
        normalize_proprio=True,
        use_image_aug=False,
        use_stronger_image_aug=False,
    )

    print(f"Statistics available at: {data_dir / 'dataset_statistics_lerobot_bi_flexiv.json'}")


if __name__ == "__main__":
    main()
