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



import numpy as np # pyright: ignore[reportMissingImports]  
from scipy.spatial.transform import Rotation # pyright: ignore[reportMissingImports]

from deepdct.geometry.se3 import project_to_so3, rotation_sqrt


# ---------------------------------------------------------
# Main label generation
# ---------------------------------------------------------

def generate_labels(T_list):

    labels = []

    for i in range(len(T_list) - 1):

        Ti = T_list[i]
        Tj = T_list[i + 1]

        Ri = project_to_so3(Ti[:3, :3])
        Rj = project_to_so3(Tj[:3, :3])

        ti = Ti[:3, 3]
        tj = Tj[:3, 3]

        # -----------------------------------------
        # Relative translation (world)
        # -----------------------------------------
        delta_t = tj - ti

        # -----------------------------------------
        # Relative rotation
        # -----------------------------------------
        Rc = Ri.T @ Rj
        Rc = project_to_so3(Rc)

        # -----------------------------------------
        # Half rotation
        # -----------------------------------------
        Rc_half = rotation_sqrt(Rc)

        # -----------------------------------------
        # Camera orientation
        #
        # Rcam = Ri @ Rc = Rj
        # -----------------------------------------
        Rcam = Rj

        # -----------------------------------------
        # Directional translation
        # -----------------------------------------
        tc = Rcam.T @ Rc_half @ delta_t

        # -----------------------------------------
        # Euler labels
        # -----------------------------------------
        euler = Rotation.from_matrix(Rc).as_euler(
            "xyz",
            degrees=False
        )

        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        labels.append([
            tc[0],
            tc[1],
            tc[2],
            roll,
            pitch,
            yaw,
        ])

    return np.asarray(labels)