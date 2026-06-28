import numpy as np
from scipy.spatial.transform import Rotation

from deepdct.geometry.se3 import rotation_sqrt


def test_rotation_sqrt_identity():
    R = np.eye(3)

    R_half = rotation_sqrt(R)

    assert np.allclose(R_half, np.eye(3), atol=1e-8)


def test_rotation_sqrt_reconstructs_rotation():
    R = Rotation.from_euler(
        "xyz",
        [0.1, -0.05, 0.2],
        degrees=False,
    ).as_matrix()

    R_half = rotation_sqrt(R)

    reconstructed = R_half @ R_half

    assert np.allclose(reconstructed, R, atol=1e-6)


def test_rotation_sqrt_is_valid_rotation():
    R = Rotation.from_euler(
        "z",
        0.3,
        degrees=False,
    ).as_matrix()

    R_half = rotation_sqrt(R)

    assert np.allclose(R_half.T @ R_half, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R_half), 1.0, atol=1e-6)