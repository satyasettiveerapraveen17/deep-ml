import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Jacobian matrix using numerical differentiation.

    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences

    Returns:
        Jacobian matrix as list of lists
    """

    # Convert x and f(x) to NumPy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(f(x), dtype=float)

    j = np.zeros((len(y), len(x)))

    for i in range(len(x)):
        x_perturbed = x.copy()
        x_perturbed[i] += h

        y_perturbed = np.asarray(f(x_perturbed), dtype=float)

        # Store derivatives in column i
        j[:, i] = (y_perturbed - y) / h

    return j.tolist()