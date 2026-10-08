import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Your code here
    eignv , V = np.linalg.eigh(A.T @ A)
    ind = np.argsort(eignv)[: : -1]
    V = V[:,ind]
    S = np.sqrt(np.maximum(eignv[ind], 0))

    evals_u, U = np.linalg.eigh(A @ A.T)
    idx_u = np.argsort(evals_u)[::-1] 
    U = U[:, idx_u]

    for i in range(len(S)):
        if S[i] > 1e-12 and np.dot(U[:, i], A @ V[:, i]) < 0:
                U[:, i] *= -1

    Vt = V.T
    return (U, S, Vt)