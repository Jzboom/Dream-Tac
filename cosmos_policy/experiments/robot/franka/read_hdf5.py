# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Read and print structure of a preprocessed Franka HDF5 episode.

Usage:
    python cosmos_policy/experiments/robot/franka/read_hdf5.py [path.hdf5]
    python cosmos_policy/experiments/robot/franka/read_hdf5.py /share/project/yunfan/tactile_data_preprocessed_whiteboard/pick_eraser_and_erase_marker_from_whiteboard/train/episode_0.hdf5
"""

import argparse
import sys

import h5py
import numpy as np


def print_structure(name, obj, indent=0):
    prefix = "  " * indent
    if isinstance(obj, h5py.Dataset):
        arr = obj[()]
        if arr.dtype.kind in ("S", "U", "O"):
            try:
                val = arr.tobytes().decode("utf-8", errors="replace").strip("\x00")
                print(f"{prefix}{name}: str = {repr(val)[:70]}")
            except Exception:
                print(f"{prefix}{name}: shape={np.asarray(arr).shape} dtype={arr.dtype}")
        else:
            arr = np.asarray(arr)
            print(f"{prefix}{name}: shape={arr.shape} dtype={arr.dtype}")
            if arr.size > 0 and arr.ndim == 2:
                print(f"{prefix}  first: {arr[0]}")
                if arr.shape[0] > 1:
                    print(f"{prefix}  last:  {arr[-1]}")
    elif isinstance(obj, h5py.Group):
        print(f"{prefix}{name}/")
        for k in sorted(obj.keys()):
            print_structure(k, obj[k], indent + 1)


def main():
    parser = argparse.ArgumentParser(description="Read preprocessed Franka HDF5 and print structure.")
    parser.add_argument(
        "path",
        nargs="?",
        default="/share/project/yunfan/hupai/hupai/hupai_20260309/2.hdf5",
        help="Path to .hdf5 file",
    )
    args = parser.parse_args()

    try:
        with h5py.File(args.path, "r") as f:
            print(f"# {args.path}\n")
            if f.attrs:
                print("attrs:")
                for k, v in f.attrs.items():
                    if isinstance(v, bytes):
                        v = v.decode("utf-8")
                    print(f"  {k}: {v}")
                print()

            print("structure:")
            for key in sorted(f.keys()):
                print_structure(key, f[key], indent=0)

            print("\n--- Summary ---")
            qpos = f["observations/qpos"][:]
            action = f["action"][:]
            print(f"qpos:   T={qpos.shape[0]}, dim={qpos.shape[1]}")
            print(f"action: T={action.shape[0]}, dim={action.shape[1]}")
            if "relative_action" in f:
                rel = f["relative_action"][:]
                print(f"relative_action: T={rel.shape[0]}, dim={rel.shape[1]}")
            for key in ("cam_front", "cam_high"):
                path = f"observations/video_paths/{key}"
                if path in f:
                    v = f[path][()].tobytes().decode("utf-8", errors="replace").strip("\x00")
                    print(f"  {key}: {v}")
    except FileNotFoundError:
        print(f"File not found: {args.path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
