# orthogonality-matrix-mini-project
Mini project demonstrating orthogonality matrix concepts and 3d graphics

> How does a 3D object rotate on screen without getting distorted?
> The answer is orthogonal matrices.

---

## What this project demonstrates

When a 3D object rotates on screen — in a game, an app, anywhere — the computer is doing matrix multiplication behind the scenes. **Orthogonal matrices** are the reason the object looks correct and doesn't get stretched or squished.

We take one 3D cube and put it through 4 steps to prove this:

**Step 1 — Show the difference visually**
Rotate the cube with an orthogonal matrix → it stays a cube.
Scale it with a non-orthogonal matrix → it gets stretched.
This is the "why does orthogonality matter" moment.

**Step 2 — Prove the rotation matrix is valid**
Run row reduction on it. Rank comes out as 3 — meaning it covers full 3D space and doesn't collapse any dimension. A rank less than 3 would mean the cube gets flattened.

**Step 3 — Flatten 3D onto a 2D screen**
Use the formula P = A(AᵀA)⁻¹Aᵀ to project the rotated cube onto a flat plane — exactly what a camera does. Then fit a line through the projected points using least squares (same formula, different use).

**Step 4 — Prove WHY nothing stretches (eigenvalues)**
Find the eigenvalues of the rotation matrix. Every single one has |λ| = 1. Eigenvalues tell you how much a matrix stretches each direction — if all are 1, nothing gets stretched. The eigenvector with λ=1 is the rotation axis — the one point that doesn't move.

---

## Team

| Topic | File | Author | What it covers |
|-------|------|--------|----------------|
| 1 | `orthogonality.py` | namratha m | Matrix representation, orthogonality check, rotation vs scaling |
| 2 | `rank_basis.py` | namratha m | Row reduction, rank, basis, null space |
| 3 | `projections_least_squares.py` | Raakshit SK | Orthogonal projection onto a plane, least squares fit |
| 4 | `eigen_rotation.py` | priyamvada| Eigenvalue analysis of rotation matrices |

---

## How to run

```bash
# install dependencies (only needed once)
pip install numpy matplotlib

# run the full project
python main.py
```

---

## Core formula

Both orthogonal projection and least squares come from the same expression:

```
P = A (AᵀA)⁻¹ Aᵀ
```

- Used as a **projection matrix** → flattens 3D points onto a plane (the screen)
- Used in **least squares** → finds the best-fit line through data points

---

## Key results

- Rotation matrix: orthogonal ✓ | det = +1 ✓ | rank = 3 ✓
- All eigenvalues: |λ| = 1 ✓ (nothing gets stretched)
- Projection: P² = P ✓ (project twice = same result)
