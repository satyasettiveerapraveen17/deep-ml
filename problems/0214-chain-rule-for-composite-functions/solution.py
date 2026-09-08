import numpy as np


func_map = {
    'square': (lambda val: val**2, lambda val: 2 * val),
    'sin': (lambda val: np.sin(val), lambda val: np.cos(val)),
    'exp': (lambda val: np.exp(val), lambda val: np.exp(val)),
    'log': (lambda val: np.log(val), lambda val: 1 / val)
}

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.
    
    Args:
        functions: List of function names (applied right to left)
                  Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative
    
    Returns:
        Derivative value at x
    
    Example:
        ['sin', 'square'] represents sin(x²)
        ['exp', 'sin', 'square'] represents exp(sin(x²))
    """
    total_derivative = 1.0
    current_value = x
    
    for func_name in reversed(functions):
        func, deriv_func = func_map[func_name]
        total_derivative *= deriv_func(current_value)
        current_value = func(current_value)

    return total_derivative
