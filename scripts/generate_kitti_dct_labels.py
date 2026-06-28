#!/usr/bin/env python3
"""
Generate DeepDCT-VO labels from KITTI odometry pose files.

Input:
    pose_dir/
        00.txt
        ...
        10.txt

Output:
    output_dir/
        00_dct.txt
        ...
        10_dct.txt

Each output row contains:

tx ty tz roll pitch yaw

where

(tx,ty,tz) = directional translation
(roll,pitch,yaw) = relative Euler angles
"""

import argparse
from deepdct.data.label_io import process_directory

# ---------------------------------------------------------
# Entry
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing KITTI pose files.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save DCT labels.",
    )

    args = parser.parse_args()

    process_directory(
        args.input_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    main()