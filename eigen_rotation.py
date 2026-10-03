"""
eigen_rotation.py
-----------------
Eigen analysis of a rotation matrix (2D or 3D).

Main entry point (for team integration):

    from eigen_rotation import eigen_analysis_of_rotation
    result = eigen_analysis_of_rotation(theta=np.pi / 3)             # 2D
    result = eigen_analysis_of_rotation(theta=np.pi / 3, axis=[0, 0, 1])  # 3D

Theory summary
--------------
2D rotation R(theta) = [[cos t, -sin t], [sin t, cos t]]
    characteristic polynomial: lambda^2 - 2 cos(t) lambda + 1 = 0
    eigenvalues: e^{+it}, e^{-it} (complex unless t = 0 or pi)
    eigenvectors: (1, -i)/sqrt2  and  (1, i)/sqrt2

3D rotation by theta about unit axis u
    eigenvalues: 1, e^{+it}, e^{-it}
    eigenvector for eigenvalue 1 is the rotation axis u
    (det = 1, R^T R = I  -> R is orthogonal, all |lambda| = 1)
"""

import numpy as np


# --------------------------------------------------------------------------
# Building rotation matrices
# --------------------------------------------------------------------------
def rotation_matrix_2d(theta):
    """2D rotation matrix for angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def rotation_matrix_3d(theta, axis):
    """3D rotation matrix about `axis` by theta radians (Rodrigues formula)."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Rotation axis must be a non-zero vector.")
    x, y, z = axis / norm
    K = np.array([[0, -z,  y],
                  [z,  0, -x],
                  [-y, x,  0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


# --------------------------------------------------------------------------
# Main callable
# --------------------------------------------------------------------------
def eigen_analysis_of_rotation(theta=None, axis=None, matrix=None,
                               verbose=True, plot=False, tol=1e-9):
    """
    Perform eigen analysis of a rotation.

    Provide EITHER `matrix` (a 2x2 or 3x3 rotation matrix)
    OR `theta` (radians), with optional `axis` for 3D.

    Parameters
    ----------
    theta   : float, rotation angle in radians
    axis    : 3-vector for a 3D rotation axis (omit for 2D)
    matrix  : ndarray, an existing rotation matrix to analyse
    verbose : print a readable report
    plot    : (2D only) plot a vector before/after rotation
    tol     : numerical tolerance for checks

    Returns
    -------
    dict with keys:
        matrix, dimension, is_orthogonal, determinant,
        eigenvalues, eigenvectors (columns), moduli, angles_deg,
        rotation_axis (3D only), residuals (||Av - lv|| for each pair),
        verified (bool)
    """
    # ---- build / validate the matrix ----
    if matrix is not None:
        R = np.asarray(matrix, dtype=float)
        if R.ndim != 2 or R.shape[0] != R.shape[1] or R.shape[0] not in (2, 3):
            raise ValueError("matrix must be 2x2 or 3x3.")
    elif theta is not None:
        R = rotation_matrix_3d(theta, axis) if axis is not None else rotation_matrix_2d(theta)
    else:
        raise ValueError("Provide either `matrix` or `theta`.")

    n = R.shape[0]
    is_orthogonal = np.allclose(R.T @ R, np.eye(n), atol=tol)
    det = float(np.linalg.det(R))
    if not (is_orthogonal and abs(det - 1) < 1e-6):
        raise ValueError("Input is not a proper rotation matrix "
                         "(need R^T R = I and det R = +1).")

    # ---- eigen decomposition ----
    eigvals, eigvecs = np.linalg.eig(R)

    # sort: real eigenvalue (1) first, then by angle
    order = np.lexsort((-np.angle(eigvals), np.abs(eigvals.imag) > tol))
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    residuals = np.array([np.linalg.norm(R @ eigvecs[:, i] - eigvals[i] * eigvecs[:, i])
                          for i in range(n)])
    moduli = np.abs(eigvals)
    angles_deg = np.degrees(np.angle(eigvals))

    result = {
        "matrix": R,
        "dimension": n,
        "is_orthogonal": is_orthogonal,
        "determinant": det,
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "moduli": moduli,
        "angles_deg": angles_deg,
        "residuals": residuals,
        "verified": bool(np.all(residuals < 1e-8) and np.allclose(moduli, 1, atol=1e-8)),
    }

    # ---- 3D: recover rotation axis & angle ----
    if n == 3:
        idx = int(np.argmin(np.abs(eigvals - 1)))
        axis_vec = np.real(eigvecs[:, idx])
        axis_vec = axis_vec / np.linalg.norm(axis_vec)
        angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        result["rotation_axis"] = axis_vec
        result["rotation_angle_deg"] = float(angle)

    if verbose:
        _print_report(result)
    if plot and n == 2:
        _plot_2d(R)

    return result


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _fmt_c(z):
    z = complex(z)
    return f"{z.real:+.4f}{z.imag:+.4f}j"


def _print_report(res):
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 56)
    print(f" EIGEN ANALYSIS OF {res['dimension']}D ROTATION")
    print("=" * 56)
    print("Rotation matrix R:\n", res["matrix"])
    print(f"\nOrthogonal (R^T R = I): {res['is_orthogonal']}")
    print(f"det(R) = {res['determinant']:.4f}")
    print("\nEigenvalues:")
    for i, lam in enumerate(res["eigenvalues"]):
        print(f"  lambda_{i+1} = {_fmt_c(lam)}   |lambda| = {res['moduli'][i]:.4f}"
              f"   arg = {res['angles_deg'][i]:+.2f} deg")
    print("\nEigenvectors (columns):")
    for i in range(res["dimension"]):
        v = "  ".join(_fmt_c(x) for x in res["eigenvectors"][:, i])
        print(f"  v_{i+1} = [ {v} ]")
    if res["dimension"] == 3:
        print(f"\nRotation axis (eigenvector for lambda=1): {res['rotation_axis']}")
        print(f"Rotation angle: {res['rotation_angle_deg']:.2f} deg")
    print("\nResiduals ||Av - lambda v||:", res["residuals"])
    print("All checks passed:", res["verified"])
    print("=" * 56)


def _plot_2d(R):
    import matplotlib.pyplot as plt
    v = np.array([1.0, 0.5])
    w = R @ v
    plt.figure(figsize=(4, 4))
    plt.quiver(0, 0, *v, angles="xy", scale_units="xy", scale=1, color="blue", label="v")
    plt.quiver(0, 0, *w, angles="xy", scale_units="xy", scale=1, color="red", label="Rv")
    plt.xlim(-2, 2); plt.ylim(-2, 2)
    plt.gca().set_aspect("equal"); plt.grid(True); plt.legend()
    plt.title("No real eigenvector: every vector turns")
    plt.show()


# --------------------------------------------------------------------------
# Run standalone to test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    eigen_analysis_of_rotation(theta=np.pi / 3)                          # 2D, 60 deg
    eigen_analysis_of_rotation(theta=np.pi / 4, axis=[1, 1, 1])          # 3D