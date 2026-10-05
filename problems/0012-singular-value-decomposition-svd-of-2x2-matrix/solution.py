import numpy as np

def custom_svd(A):
    
    # 1. Compute V and singular values squared from A^T * A
    ATA = np.dot(A.T, A)
    eigenvalues_V, V = np.linalg.eigh(ATA)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices_V = np.argsort(eigenvalues_V)[::-1]
    eigenvalues_V = eigenvalues_V[sorted_indices_V]
    V = V[:, sorted_indices_V]
    
    # Singular values are the square roots of the eigenvalues
    # Clip at 0 to avoid tiny negative numbers due to floating-point precision
    singular_values = np.sqrt(np.clip(eigenvalues_V, 0, None))
    
    # 2. Compute U from A * A^T
    AAT = np.dot(A, A.T)
    eigenvalues_U, U = np.linalg.eigh(AAT)
    
    # Sort U in descending order
    sorted_indices_U = np.argsort(eigenvalues_U)[::-1]
    U = U[:, sorted_indices_U]
    
    # 3. Fix sign consistency (Signs of U and V must align so U * Sigma * V^T = A)
    for i in range(len(singular_values)):
        if singular_values[i] > 1e-9:
            # Recompute column of U based on V to guarantee sign matching
            # u_i = A * v_i / sigma_i
            U[:, i] = np.dot(A, V[:, i]) / singular_values[i]
            
    return U, singular_values, V.T

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
    return  custom_svd(A)