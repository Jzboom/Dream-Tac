# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precompute T5 text embeddings for LeRobot datasets with meta/tasks.parquet."""

from __future__ import annotations

import argparse
import os

import pandas as pd

from cosmos_policy.datasets.t5_embedding_utils import generate_t5_embeddings, save_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute T5 text embeddings from LeRobot tasks.parquet")
    parser.add_argument("--data_dir", type=str, required=True, help="LeRobot dataset directory")
    parser.add_argument(
        "--t5_model_path",
        type=str,
        default="google-t5/t5-11b",
        help="Local T5 directory or HuggingFace model id. Use a local directory unless --allow_download is set.",
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional HuggingFace cache directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device for T5 encoding, e.g. cuda or cuda:0")
    parser.add_argument("--allow_download", action="store_true", help="Allow transformers to download the T5 model")
    args = parser.parse_args()

    tasks_path = os.path.join(args.data_dir, "meta", "tasks.parquet")
    if not os.path.exists(tasks_path):
        raise FileNotFoundError(f"tasks.parquet not found: {tasks_path}")

    tasks = pd.read_parquet(tasks_path)
    text_col = "task" if "task" in tasks.columns else "tasks" if "tasks" in tasks.columns else None
    if text_col is None:
        unique_commands = [str(index) for index in tasks.index]
    else:
        unique_commands = sorted({str(command) for command in tasks[text_col].dropna().tolist()})
    print(f"Task name(s): {unique_commands}")
    t5_text_embeddings = generate_t5_embeddings(
        unique_commands,
        model_name=args.t5_model_path,
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_download,
    )
    save_path = save_embeddings(t5_text_embeddings, args.data_dir)
    print(f"Done. Add to experiment config: t5_text_embeddings_path=\"{save_path}\"")


if __name__ == "__main__":
    main()
