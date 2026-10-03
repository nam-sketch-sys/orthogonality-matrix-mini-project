"""
Topic 1: Matrix representation & orthogonality
Topic 2: Simplification (row reduction), rank & basis

Modular: main.py does `import topic1_2_matrix_rank_basis as t12`
and calls t12.run_topic1() and t12.run_topic2()  (or t12.run_demo() for both)
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
# TOPIC 2: SIMPLIFICATION, RANK & BASIS  (built from scratch with row reduction)
# ======================================================================

def rref(A):
    """
    Reduced row echelon form using Gauss-Jordan elimination.
    Returns (R, pivot_columns).
    """
    R = np.array(A, dtype=float)
    rows, cols = R.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        # pick the row with the largest entry in this column (partial pivoting)
        p = r + np.argmax(np.abs(R[r:, c]))
        if abs(R[p, c]) < TOL:
            continue  # no pivot in this column
        R[[r, p]] = R[[p, r]]          # swap rows
        R[r] = R[r] / R[r, c]          # make pivot = 1
        for i in range(rows):          # clear the rest of the column
            if i != r:
                R[i] = R[i] - R[i, c] * R[r]
        pivots.append(c)
        r += 1
    R[np.abs(R) < TOL] = 0.0
    return R, pivots


def rref_steps(A):
    """Simplify A to RREF and PRINT every row operation (good for the viva)."""
    R = np.array(A, dtype=float)
    rows, cols = R.shape
    r = 0
    print("Start:\n", np.round(R, 4))
    for c in range(cols):
        if r >= rows:
            break
        p = r + np.argmax(np.abs(R[r:, c]))
        if abs(R[p, c]) < TOL:
            continue
        if p != r:
            R[[r, p]] = R[[p, r]]
            print(f"\nSwap R{r+1} <-> R{p+1}:\n", np.round(R, 4))
        piv = R[r, c]
        R[r] = R[r] / piv
        print(f"\nR{r+1} -> R{r+1} / {piv:.4g}:\n", np.round(R, 4))
        for i in range(rows):
            if i != r and abs(R[i, c]) > TOL:
                f = R[i, c]
                R[i] = R[i] - f * R[r]
                print(f"\nR{i+1} -> R{i+1} - ({f:.4g}) R{r+1}:\n", np.round(R, 4))
        r += 1
    R[np.abs(R) < TOL] = 0.0
    return R


def inverse_gauss_jordan(A):
    """Simplify [A | I] to [I | A^-1]. Returns None if A is singular."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    R, piv = rref(np.hstack([A, np.eye(n)]))
    if piv[:n] != list(range(n)):
        return None
    return R[:, n:]


def solve_system(A, b):
    """Solve Ax = b by simplifying the augmented matrix [A | b] and comparing ranks."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    rA = rank(A)
    rAb = rank(np.hstack([A, b]))
    n = A.shape[1]
    if rA < rAb:
        return "No solution (rank A < rank [A|b])", None
    R, piv = rref(np.hstack([A, b]))
    if rA == n:
        return "Unique solution", R[:n, -1]
    return f"Infinitely many solutions ({n - rA} free variable(s))", R[:rA, -1]


def rank(A):
    return len(rref(A)[1])


def column_space_basis(A):
    """Basis of Col(A) = the ORIGINAL columns of A at the pivot positions."""
    A = np.array(A, dtype=float)
    _, piv = rref(A)
    return A[:, piv]


def row_space_basis(A):
    """Basis of Row(A) = the non-zero rows of the RREF."""
    R, piv = rref(A)
    return R[:len(piv)]


def null_space_basis(A):
    """Basis of Null(A): one vector per free variable."""
    A = np.array(A, dtype=float)
    R, piv = rref(A)
    n = A.shape[1]
    free = [c for c in range(n) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(n)
        v[f] = 1.0
        for row, pc in enumerate(piv):
            v[pc] = -R[row, f]
        basis.append(v)
    return np.array(basis).T if basis else np.zeros((n, 0))


def are_independent(vectors):
    """Columns are linearly independent <=> rank = number of vectors."""
    M = np.array(vectors, dtype=float)
    return rank(M) == M.shape[1]


def rank_report(name, A):
    A = np.array(A, dtype=float)
    R, piv = rref(A)
    r = len(piv)
    n = A.shape[1]
    print(f"\n=== {name} ===")
    print("A =\n", A)
    print("RREF =\n", np.round(R, 4))
    print("Pivot columns:", piv)
    print("Rank =", r, "| numpy check:", np.linalg.matrix_rank(A))
    print("Basis of column space (columns):\n", np.round(column_space_basis(A), 4))
    print("Basis of row space (rows):\n", np.round(row_space_basis(A), 4))
    N = null_space_basis(A)
    print("Basis of null space (columns):\n", np.round(N, 4))
    print(f"Rank-Nullity: rank {r} + nullity {n - r} = {n} columns ->",
          "OK" if r + (n - r) == n else "ERROR")
    print("A @ null vectors ~ 0 :", np.allclose(A @ N, 0))


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


def run_topic2():
    print("\n########## TOPIC 2: SIMPLIFICATION, RANK & BASIS ##########")
    A = [[1, 2, 3],
         [2, 4, 6],
         [1, 0, 1]]

    print("\n>>> Step-by-step simplification (RREF) of A")
    rref_steps(A)

    rank_report("Rank-deficient matrix (row 2 = 2 x row 1)", A)

    B = [[1, 0, 2, 1],
         [0, 1, 1, 3],
         [1, 1, 3, 4]]
    rank_report("3x4 matrix (has non-trivial null space)", B)

    print("\n>>> Inverse by Gauss-Jordan on [A | I]")
    M = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
    inv = inverse_gauss_jordan(M)
    print(np.round(inv, 4))
    print("Matches numpy inverse:", np.allclose(inv, np.linalg.inv(M)))
    print("Singular matrix inverse ->", inverse_gauss_jordan(A))

    print("\n>>> Solving Ax = b using ranks")
    print(solve_system([[1, 1], [1, -1]], [3, 1]))
    print(solve_system([[1, 1], [2, 2]], [3, 6]))
    print(solve_system([[1, 1], [2, 2]], [3, 7]))

    print("\nAre (1,0,0),(0,1,0),(1,1,0) independent? ->",
          are_independent(np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]]).T))
    print("Are columns of a rotation matrix independent? ->", are_independent(rotation_z(0.5)))


def run_demo(show_plots=True):
    run_topic1(show_plots)
    run_topic2()


if __name__ == "__main__":
    run_demo()