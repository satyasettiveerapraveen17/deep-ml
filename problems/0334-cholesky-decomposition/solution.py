import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    # 1. Convert input to a numpy array
    A = np.array(A, dtype=float)
    
    # 2. Check for empty inputs or non-square dimensions
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        return -1
        
    if not np.allclose(A, A.T):
        return -1
        
    try:
        L = np.linalg.cholesky(A)
        return L.tolist()  
        
    except np.linalg.LinAlgError:
        return -1
