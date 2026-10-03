"""
Topic 3: Orthogonal Projections and Least Squares
--------------------------------------------------

Both concepts use the same core formula  P = A (A^T A)^-1 A^T
  - Projection : drops a 3D point onto a plane (like a shadow)
  - Least Squares : finds the best-fit line through noisy data
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ------------------------------------------------------------------
# Core formula — P = A (A^T A)^-1 A^T
# ------------------------------------------------------------------

def projection_matrix(A):
    A = np.asarray(A, dtype=float)
    return A @ np.linalg.inv(A.T @ A) @ A.T


def project_points_onto_plane(points, plane_basis):
    P = projection_matrix(plane_basis)
    return (P @ points.T).T


# x_hat = (A^T A)^-1 A^T b
def least_squares_fit(A, b):
    x_hat = np.linalg.inv(A.T @ A) @ A.T @ b
    b_fit  = A @ x_hat
    return x_hat, b_fit


# ------------------------------------------------------------------
# Demo — called by main.py
# ------------------------------------------------------------------

def run_topic3(show_plots=True):
    print("\n########## TOPIC 3: ORTHOGONAL PROJECTIONS & LEAST SQUARES ##########")

    # ── Part A : Projection ───────────────────────────────────────
    print("\n--- Orthogonal Projection onto the XY-plane ---")
    print("Think of it as: shine a light straight down, look at the shadow on the floor.\n")

    # ✏️  Change these points to see different shadows
    points = np.array([
        [1, 2, 3],
        [4, 0, 1],
        [2, 5, 2],
        [3, 3, 3],
        [0, 1, 4]
    ], dtype=float)

    # XY-plane: spanned by [1,0,0] and [0,1,0]  (Z is dropped)
    xy_basis = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ], dtype=float)

    P = projection_matrix(xy_basis)
    print("Projection matrix P (3x3):\n", np.round(P, 4))
    print("\nKey check — P squared should equal P  (P^2 = P):",
          np.allclose(P @ P, P))

    flattened = project_points_onto_plane(points, xy_basis)
    print("\nOriginal 3D points      →  Projected (shadow):")
    for orig, proj in zip(points, flattened):
        print(f"  {orig}  →  {np.round(proj, 4)}")

    # ── Part B : Least Squares ────────────────────────────────────
    print("\n--- Least Squares Line Fit ---")
    print("Finding the best-fit line y = mt + c through noisy data.\n")

    np.random.seed(0)
    t         = np.linspace(0, 10, 20)
    true_line = 2 * t + 1                                   # ✏️  change slope/intercept here
    noisy     = true_line + np.random.normal(0, 1.5, size=t.shape)

    # Build the model matrix A  (each row is [t_i, 1])
    A = np.column_stack([t, np.ones_like(t)])

    x_hat, fit_vals = least_squares_fit(A, noisy)
    print(f"True line   : y = 2.000 t + 1.000")
    print(f"Fitted line : y = {x_hat[0]:.3f} t + {x_hat[1]:.3f}")
    print(f"\nError (||true - fit||): {np.linalg.norm(true_line - fit_vals):.4f}")

    # ── Plots ─────────────────────────────────────────────────────
    if not show_plots:
        return

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("Topic 3 — Orthogonal Projections & Least Squares", fontsize=13)

    # Panel 1: 3D shadow
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(*points.T,   color="red",  s=60, label="Original 3D points")
    ax1.scatter(*flattened.T, color="blue", s=60, label="Shadow on XY-plane")
    for p, f in zip(points, flattened):
        ax1.plot(*zip(p, f), "k--", alpha=0.35)
    ax1.set_title("Orthogonal Projection\n(P = A(AᵀA)⁻¹Aᵀ)")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.legend()

    # Panel 2: least squares
    ax2 = fig.add_subplot(122)
    ax2.scatter(t, noisy,     color="gray",  label="Noisy data", zorder=3)
    ax2.plot(t, fit_vals,     color="green", linewidth=2,
             label=f"Fit: y = {x_hat[0]:.2f}t + {x_hat[1]:.2f}")
    ax2.plot(t, true_line, "r--", label="True line (y = 2t + 1)")
    ax2.set_title("Least Squares Fit\n(x̂ = (AᵀA)⁻¹Aᵀb)")
    ax2.set_xlabel("t"); ax2.set_ylabel("y")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("topic3_projections_least_squares.png", dpi=150)
    print("\nPlot saved → topic3_projections_least_squares.png")
    plt.show()


if __name__ == "__main__":
    run_topic3()
