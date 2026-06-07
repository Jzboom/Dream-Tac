"""
Check which HDF5 files have 2D (or 1D) puppet/joint.
Usage:
    python cosmos_policy/experiments/robot/franka/check_joint_shapes.py [--dataset_path PATH]
"""
import argparse
import os

import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default="/share/project/yunfan/tactile_data/pick_eraser_and_erase_marker_from_whiteboard",
        help="Root dir to search for .hdf5 files",
    )
    parser.add_argument("--show_2d_only", action="store_true", help="Only print files where joint is 2D")
    args = parser.parse_args()

    paths = []
    for root_dir, _dirs, files in os.walk(args.dataset_path):
        for f in files:
            if f.endswith(".hdf5"):
                paths.append(os.path.join(root_dir, f))
    paths.sort()

    ndim_1 = []
    ndim_2 = []
    other = []

    for p in paths:
        try:
            with h5py.File(p, "r") as f:
                joint = f["puppet/joint"]
                sh = joint.shape
                nd = joint.ndim
                if nd == 1:
                    ndim_1.append((p, sh))
                elif nd == 2:
                    ndim_2.append((p, sh))
                else:
                    other.append((p, sh))
        except Exception as e:
            print(f"Error {p}: {e}")

    if args.show_2d_only:
        print("Files where joint is 2D:")
        for p, sh in ndim_2:
            print(f"  {p}  shape={sh}")
        return

    print("Joint ndim == 1:")
    for p, sh in ndim_1:
        print(f"  {p}  shape={sh}")
    print("\nJoint ndim == 2:")
    for p, sh in ndim_2:
        print(f"  {p}  shape={sh}")
    if other:
        print("\nOther ndim:")
        for p, sh in other:
            print(f"  {p}  shape={sh}")
    print(f"\nSummary: 1D={len(ndim_1)}, 2D={len(ndim_2)}, other={len(other)}, total={len(paths)}")


if __name__ == "__main__":
    main()
