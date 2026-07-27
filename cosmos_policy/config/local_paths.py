"""Default paths for the repository's sibling-directory deployment layout.

Expected layout::

    workspace/
    ├── Dream-Tac/
    ├── pick_up_cube_0713/
    └── checkpoints/
        └── Cosmos-Predict2-2B-Video2World/

Paths are anchored to this file rather than the process working directory.
"""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent

DEFAULT_LEROBOT_DATA_DIR = WORKSPACE_ROOT / "pick_up_cube_0713"
DEFAULT_CHECKPOINT_ROOT = WORKSPACE_ROOT / "checkpoints"
DEFAULT_COSMOS_PREDICT2_ROOT = DEFAULT_CHECKPOINT_ROOT / "Cosmos-Predict2-2B-Video2World"
DEFAULT_COSMOS_PREDICT2_MODEL = DEFAULT_COSMOS_PREDICT2_ROOT / "model-480p-16fps.pt"
DEFAULT_WAN_VAE = DEFAULT_COSMOS_PREDICT2_ROOT / "tokenizer" / "tokenizer.pth"
