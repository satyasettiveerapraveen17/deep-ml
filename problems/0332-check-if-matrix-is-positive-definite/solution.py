import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    matrix = np.array(matrix)
    
    # 1. Ensure the matrix is square and 2D
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return {"is_positive_definite": False, "eigenvalues": []}
    
    # 2. Check symmetry (needed for true positive definiteness math)
    is_symmetric = np.allclose(matrix, matrix.T)
        
    # 3. Compute eigenvalues and drop any tiny complex parts
    eigenvalues = np.linalg.eigvals(matrix).real
    
    # 4. Round to 4 decimal places and sort in ascending order
    sorted_eigenvalues = np.sort(np.round(eigenvalues, 4)).tolist()
    
    # 5. Check if positive definite (must be symmetric AND all eigenvalues > 0)
    is_pd = bool(is_symmetric and np.all(eigenvalues > 0))
    
    return {
        "is_positive_definite": is_pd,
        "eigenvalues": sorted_eigenvalues
    }
