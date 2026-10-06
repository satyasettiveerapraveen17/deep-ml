import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Perform QR decomposition using Gram-Schmidt process.
    
    Args:
        A: An m x n matrix represented as list of lists
    
    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular
    """
    A = np.array(A, dtype=float)
    m, n = A.shape

    # Fix 1: Pass shape as a tuple
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        # Fix 2 & 4: Grab the j-th column and use .copy()
        v = A[:, j].copy()
        
        # Fix 3: Add the missing 'in' keyword
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
            
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return (Q, R)
