import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    # 1. Compute eigenvalues
    eign_values = np.linalg.eigvals(hessian)
    
    # 2. Check if any eigenvalue is effectively zero using the tolerance
    if np.any(np.abs(eign_values) < tol):
        return None
        
    # 3. Check if all eigenvalues are strictly positive
    elif np.all(eign_values > 0):
        return -1
        
    # 4. Check if all eigenvalues are strictly negative
    elif np.all(eign_values < 0):
        return 1
        
    # 5. Mixed positive and negative (Saddle point)
    return 0
