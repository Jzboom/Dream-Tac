"""
Read puppet data (gripper, joint, pose) from tactile_data HDF5 files.
"""
import argparse
import h5py
import numpy as np


def read_puppet(hdf5_path: str):
    """
    Read puppet/gripper, puppet/joint, puppet/pose from HDF5.

    Returns:
        dict with keys: gripper (T, 2), joint (T, 7), pose (T, 6)
    """
    with h5py.File(hdf5_path, "r") as f:
        gripper = np.array(f["puppet/gripper"][:], dtype=np.float32)
        joint = np.array(f["puppet/joint"][:], dtype=np.float32)
        pose = np.array(f["puppet/pose"][:], dtype=np.float32)
    return {"gripper": gripper, "joint": joint, "pose": pose}


def main():
    parser = argparse.ArgumentParser(description="Read puppet data from tactile HDF5")
    parser.add_argument("--hdf5_path", help="Path to HDF5 file", default="/share/project/yunfan/tactile_data/insert_nut_into_screw/insert_screw_202602010_03/2.hdf5")
    parser.add_argument("--frame", type=int, default=None, metavar="N", help="Print single frame (0-indexed)")
    args = parser.parse_args()

    data = read_puppet(args.hdf5_path)
    gripper, joint, pose = data["gripper"], data["joint"], data["pose"]

    print(f"Loaded from {args.hdf5_path}")
    print(f"  gripper: {gripper.shape}")
    print(f"  joint:   {joint.shape}")
    print(f"  pose:    {pose.shape}")

    if args.frame is not None:
        i = args.frame
        print(f"\nFrame {i}:")
        print(f"  gripper: {gripper[i]}")
        print(f"  joint:   {joint[i]}")
        print(f"  pose:    {pose[i]}")
    else:
        print(f"\nFirst frame (0):")
        print(f"  gripper: {gripper[200:220]}")
        print(f"  joint:   {joint[0]}")
        print(f"  pose:    {pose[0]}")


if __name__ == "__main__":
    main()
