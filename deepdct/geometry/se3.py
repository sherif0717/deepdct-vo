import numpy as np  # pyright: ignore[reportMissingImports]
from scipy.linalg import logm, expm # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def project_to_so3(R):
    """Project a nearly-rotation matrix onto SO(3)."""
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rotation_sqrt(R):
    """
    Matrix square root for a rotation matrix.

    Rc^(1/2)=exp(0.5*log(R))
    """
    return np.real(expm(0.5 * logm(R)))


