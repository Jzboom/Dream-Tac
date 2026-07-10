# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared utilities for precomputing and visualizing T5 text embeddings.
"""

import os
import pickle
from typing import Dict, List

import torch
from tqdm import tqdm

from cosmos_policy._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder, get_text_embedding


def generate_t5_embeddings(
    unique_commands: List[str],
    model_name: str = "google-t5/t5-11b",
    device: str = "cuda",
    cache_dir: str | None = None,
    local_files_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Generate T5 text embeddings for a list of commands.

    Args:
        unique_commands: List of unique command strings
        model_name: Local T5 path or HuggingFace model id
        device: Device used for T5 encoding
        cache_dir: Optional HuggingFace cache directory
        local_files_only: If True, do not download model files

    Returns:
        Dictionary mapping command strings to their T5 embeddings (bfloat16, on CPU)
    """
    t5_text_embeddings = dict()
    print("Getting text embeddings...")
    try:
        encoder = CosmosT5TextEncoder(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        mode = "local-only" if local_files_only else "download-enabled"
        raise OSError(
            f"Could not load T5 model {model_name!r} in {mode} mode. "
            "Pass --t5_model_path /path/to/local/t5-11b, or add --allow_download if this machine can access HuggingFace."
        ) from exc

    for command in tqdm(unique_commands):
        embedding = get_text_embedding(command, encoder=encoder, device=device).to(dtype=torch.bfloat16).cpu()
        t5_text_embeddings[command] = embedding
    return t5_text_embeddings


def save_embeddings(t5_text_embeddings: Dict[str, torch.Tensor], data_dir: str, check_exists: bool = False) -> str:
    """
    Save T5 text embeddings to a pickle file.

    Args:
        t5_text_embeddings: Dictionary of embeddings to save
        data_dir: Directory where embeddings should be saved
        check_exists: If True, prompt user for new filename if file exists

    Returns:
        Path where embeddings were saved
    """
    print("Saving text embeddings...")
    save_path = os.path.join(data_dir, "t5_embeddings.pkl")
    if check_exists and os.path.exists(save_path):
        print(f"File {save_path} already exists.")
        new_filename = input("Please enter a new filename for saving the embeddings (e.g., t5_embeddings_v2.pkl): ")
        save_path = os.path.join(data_dir, new_filename)

    with open(save_path, "wb") as file:
        pickle.dump(t5_text_embeddings, file)
        print(f"Saved embeddings at: {save_path}")

    return save_path
