import numpy as np

def lu_decomposition(A: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform LU decomposition on a square matrix using Doolittle's method.
    
    Args:
        A: Square matrix as a list of lists
    
    Returns:
        tuple: (L, U) where L is lower triangular with 1s on diagonal,
               U is upper triangular, and A = L @ U
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        # L has 1s on the main diagonal
        L[i, i] = 1.0
        
        # Compute Upper triangular matrix (U) row elements
        for k in range(i, n):
            s = sum(L[i, j] * U[j, k] for j in range(i))
            U[i, k] = A[i, k] - s
            
        # Compute Lower triangular matrix (L) column elements
        for k in range(i + 1, n):
            s = sum(L[k, j] * U[j, i] for j in range(i))
            if U[i, i] == 0:
                raise ValueError("Zero pivot encountered.")
            L[k, i] = (A[k, i] - s) / U[i, i]
            
    return L, U