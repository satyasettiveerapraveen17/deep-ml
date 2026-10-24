import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Perform QR decomposition using Gram-Schmidt process.
    
    Args:
        A: An m x n matrix represented as list of lists
    
    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular
    """
    # 1. Convert input to a float numpy array
    A = np.array(A, dtype=float)
    m, n = A.shape
    
    # 2. Initialize empty matrices for Q and R
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    # 3. Apply the Classical Gram-Schmidt process column by column
    for j in range(n):
        # Start with the j-th column of A
        v = A[:, j].copy()
        
        # Subtract the projections of all previous orthogonal vectors
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
            
        # Compute the norm of the orthogonalized vector
        R[j, j] = np.linalg.norm(v)
        
        # Normalize to find the next column of Q (handle zero-vector edges)
        if R[j, j] > 1e-12:
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = v  # Kept as zero vector if column is linearly dependent
            
    # 4. Return both outputs back as Python lists of lists
    return Q.tolist(), R.tolist()
