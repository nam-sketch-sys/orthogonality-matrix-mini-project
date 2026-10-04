"""
gram_schmidt.py
Part: Gram-Schmidt Orthogonalization
Project: Orthogonal Matrices and 3D Graphics (UE25MA242A)

Idea:
    Given linearly independent vectors v1, v2, ..., vn, Gram-Schmidt builds
    an orthonormal set e1, e2, ..., en that spans the same space.

    Step 1: u1 = v1
    Step k: u_k = v_k - sum_{j<k} proj_{u_j}(v_k)
            where proj_u(v) = (v . u) / (u . u) * u
    Finally: e_k = u_k / ||u_k||   (normalize)

    The matrix Q with columns e1, ..., en is an ORTHOGONAL matrix:
        Q^T Q = I   and   Q^-1 = Q^T
    Such matrices represent rotations/reflections in 3D graphics.
"""

import numpy as np


def project(v, u):
    """Projection of vector v onto vector u: ((v.u)/(u.u)) * u"""
    return (np.dot(v, u) / np.dot(u, u)) * u


def gram_schmidt(A, verbose=True):
    """
    Apply Gram-Schmidt to the columns of matrix A.

    Parameters
    ----------
    A : (m x n) array, columns are linearly independent vectors
    verbose : print each step if True

    Returns
    -------
    Q : (m x n) array with orthonormal columns
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    U = []  # orthogonal (not yet normalized) vectors
    E = []  # orthonormal vectors

    for k in range(n):
        v = A[:, k]
        u = v.copy()

        # subtract the projection of v onto every previous u_j
        for j in range(k):
            u = u - project(v, U[j])

        # if u is (almost) zero, the vectors are linearly dependent
        if np.linalg.norm(u) < 1e-10:
            raise ValueError(
                f"Vector {k + 1} is linearly dependent on the previous ones."
            )

        e = u / np.linalg.norm(u)
        U.append(u)
        E.append(e)

        if verbose:
            print(f"Step {k + 1}:")
            print(f"  v{k + 1} = {v}")
            print(f"  u{k + 1} = {np.round(u, 4)}")
            print(f"  e{k + 1} = u{k + 1}/||u{k + 1}|| = {np.round(e, 4)}\n")

    return np.column_stack(E)


def is_orthogonal(Q, tol=1e-8):
    """Check whether Q^T Q = I (columns orthonormal)."""
    Q = np.array(Q, dtype=float)
    return np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)


def demo():
    # three linearly independent vectors in 3D (as columns)
    A = np.array([[1, 1, 0],
                  [1, 0, 1],
                  [0, 1, 1]], dtype=float)

    print("Input matrix A (columns are v1, v2, v3):")
    print(A, "\n")

    Q = gram_schmidt(A)

    print("Orthonormal matrix Q:")
    print(np.round(Q, 4), "\n")

    print("Check Q^T Q (should be the identity matrix):")
    print(np.round(Q.T @ Q, 4), "\n")

    print("Is Q orthogonal?", is_orthogonal(Q))
    print("Q^-1 equals Q^T?", np.allclose(np.linalg.inv(Q), Q.T))
    print("det(Q) =", round(np.linalg.det(Q), 4),
          "(+1 = rotation, -1 = reflection)")

    # 3D graphics link: orthogonal matrices preserve lengths
    p = np.array([2.0, 3.0, 4.0])
    p_new = Q @ p
    print("\nLength of point before:", round(np.linalg.norm(p), 4))
    print("Length of point after Q:", round(np.linalg.norm(p_new), 4))
    print("Lengths preserved:", np.isclose(np.linalg.norm(p), np.linalg.norm(p_new)))


if __name__ == "__main__":
    demo()
