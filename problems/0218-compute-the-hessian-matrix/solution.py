import numpy as np
from typing import Callable

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Hessian matrix of function f at the given point using finite differences.
    
    Args:
        f: A scalar function that takes a list of floats and returns a float
        point: The point at which to compute the Hessian (list of coordinates)
        h: Step size for finite differences (default: 1e-5)
        
    Returns:
        The Hessian matrix as a list of lists (n x n where n = len(point))
    """
    n = len(point)
    # Initialize an n x n zero matrix
    hessian = np.zeros((n, n))
    
    # 1. Evaluate the function at the base point
    f_0 = f(point)
    
    # 2. Iterate through all pairs of variables
    for i in range(n):
        for j in range(i, n):  # Start from 'i' because Hessian is symmetric (H_ij = H_ji)
            if i == j:
                # --- Diagonal elements: Second partial derivative (d²f / dx_i²) ---
                # Formula: (f(x + h) - 2*f(x) + f(x - h)) / h²
                p_plus = list(point)
                p_minus = list(point)
                
                p_plus[i] += h
                p_minus[i] -= h
                
                hessian[i, i] = (f(p_plus) - 2 * f_0 + f(p_minus)) / (h ** 2)
            else:
                # --- Off-diagonal elements: Mixed partial derivative (d²f / dx_i dx_j) ---
                # Formula: (f(x + h_i + h_j) - f(x + h_i - h_j) - f(x - h_i + h_j) + f(x - h_i - h_j)) / (4 * h²)
                p_pp = list(point)  # +h_i, +h_j
                p_pm = list(point)  # +h_i, -h_j
                p_mp = list(point)  # -h_i, +h_j
                p_mm = list(point)  # -h_i, -h_j
                
                p_pp[i] += h; p_pp[j] += h
                p_pm[i] += h; p_pm[j] -= h
                p_mp[i] -= h; p_mp[j] += h
                p_mm[i] -= h; p_mm[j] -= h
                
                mixed_deriv = (f(p_pp) - f(p_pm) - f(p_mp) + f(p_mm)) / (4 * (h ** 2))
                
                # Apply to both symmetric slots
                hessian[i, j] = mixed_deriv
                hessian[j, i] = mixed_deriv
                
    return hessian.tolist()
