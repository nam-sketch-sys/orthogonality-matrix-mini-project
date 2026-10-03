"""
Topic 1: Matrix representation & orthogonality

Modular: main.py does `import topic1_orthogonality as t1`
and calls t1.run_topic1()
Needs: numpy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt

TOL = 1e-10


# ======================================================================
# TOPIC 1: MATRIX REPRESENTATION & ORTHOGONALITY
# ======================================================================

def rotation_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def scaling(sx, sy, sz):
    return np.diag([sx, sy, sz]).astype(float)


def reflection_xy():
    """Reflection across the XY plane (z -> -z)."""
    return np.diag([1.0, 1.0, -1.0])


def is_orthogonal(Q):
    """Q is orthogonal  <=>  Q^T Q = I."""
    Q = np.asarray(Q, dtype=float)
    return Q.shape[0] == Q.shape[1] and np.allclose(Q.T @ Q, np.eye(Q.shape[0]), atol=TOL)


def analyse_matrix(name, A):
    """Print the 'orthogonal simplification' facts for a transformation matrix."""
    print(f"\n--- {name} ---")
    print(np.round(A, 4))
    if is_orthogonal(A):
        d = np.linalg.det(A)
        kind = "rotation (det = +1)" if d > 0 else "reflection (det = -1)"
        print(f"Orthogonal?  YES  -> {kind}")
        print("Key property: inverse = transpose (no heavy inversion needed)")
        print("Check  A^T == A^-1 :", np.allclose(A.T, np.linalg.inv(A)))
    else:
        print("Orthogonal?  NO   (changes lengths/angles, so inverse != transpose)")


def check_length_preserved(Q, v):
    """Orthogonal matrices preserve length and dot products."""
    v = np.asarray(v, dtype=float)
    print(f"|v| = {np.linalg.norm(v):.4f}   |Qv| = {np.linalg.norm(Q @ v):.4f}")


def cube_vertices():
    pts = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    return pts.T  # shape (3, 8): each column is one vertex


CUBE_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8)
              if np.sum(np.abs(cube_vertices()[:, i] - cube_vertices()[:, j])) == 2]


def plot_transformation(M, title="Cube transformed by M"):
    """Draw original cube (blue) and M @ cube (red) in 3D."""
    V = cube_vertices()
    W = M @ V
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i, j in CUBE_EDGES:
        ax.plot(*zip(V[:, i], V[:, j]), color="tab:blue", alpha=0.5)
        ax.plot(*zip(W[:, i], W[:, j]), color="tab:red")
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    lim = max(2, np.abs(W).max() + 0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=25, azim=-60)
    plt.show()


# ======================================================================
# DEMO  (main.py calls this)
# ======================================================================

def run_topic1(show_plots=True):
    print("########## TOPIC 1: MATRIX REPRESENTATION & ORTHOGONALITY ##########")
    theta = np.pi / 4
    Rz = rotation_z(theta)
    S = scaling(2, 1, 1)
    F = reflection_xy()
    shear = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

    analyse_matrix("Rotation about z by 45 deg", Rz)
    analyse_matrix("Reflection in XY plane", F)
    analyse_matrix("Scaling (2,1,1)", S)
    analyse_matrix("Shear", shear)

    v = np.array([1, 2, 3])
    print("\nLength preservation under rotation:")
    check_length_preserved(Rz, v)
    print("Length under scaling (not preserved):")
    check_length_preserved(S, v)

    combo = rotation_x(np.pi / 6) @ rotation_y(np.pi / 3) @ Rz
    print("\nProduct of rotations still orthogonal? ->", is_orthogonal(combo))

    if show_plots:
        plot_transformation(Rz, "Rotation (orthogonal): shape preserved")
        plot_transformation(S, "Scaling (not orthogonal): shape distorted")


if __name__ == "__main__":
    run_topic1()