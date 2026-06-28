import numpy as np

from deepdct.geometry.se3 import project_to_so3


def test_project_to_so3_returns_valid_rotation():
    R_noisy = np.array([
        [1.0, 0.01, 0.0],
        [-0.01, 0.99, 0.02],
        [0.0, -0.02, 1.01],
    ])

    R = project_to_so3(R_noisy)

    I = np.eye(3)

    assert R.shape == (3, 3)
    assert np.allclose(R.T @ R, I, atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_project_to_so3_keeps_identity_close():
    R = project_to_so3(np.eye(3))

    assert np.allclose(R, np.eye(3), atol=1e-8)