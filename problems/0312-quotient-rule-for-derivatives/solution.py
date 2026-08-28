import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    # Your code here
    g = np.poly1d(g_coeffs)
    h = np.poly1d(h_coeffs)

    # Derivatives
    g_prime = np.polyder(g)
    h_prime = np.polyder(h)

    # Quotient rule
    numerator = g_prime(x) * h(x) - g(x) * h_prime(x)
    denominator = h(x) ** 2

    return numerator / denominator
