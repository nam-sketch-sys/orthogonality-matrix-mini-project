import numpy as np
import orthogonality as t1
import rank_basis as t2
import eigen_rotation as t5
from gram_schmidt import gram_schmidt
t1.run_topic1()
t2.run_topic2()
t5.eigen_analysis_of_rotation(theta=3.14159/3)
t5.eigen_analysis_of_rotation(theta=3.14159/4, axis=[1, 1, 1])
print("\n--- Gram-Schmidt Orthogonalization ---")
A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

Q = gram_schmidt(A)

print("Orthonormal matrix Q:")
print(np.round(Q, 4))
print("Q^T Q = I ?", np.allclose(Q.T @ Q, np.eye(3)))
