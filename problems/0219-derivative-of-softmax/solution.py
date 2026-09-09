import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
    """
    Compute the Jacobian matrix of the softmax function.

    Args:
        x: Input vector of real numbers

    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i) / d(x_j)
    """

    x = np.array(x, dtype=float)

    # Softmax
    exp_x = np.exp(x - np.max(x))
    softmax = exp_x / np.sum(exp_x)

    # Jacobian matrix
    J = np.zeros((len(x), len(x)))

    for i in range(len(x)):
        for k in range(len(x)):
            if i == k:
                J[i, k] = softmax[i] * (1 - softmax[i])
            else:
                J[i, k] = -softmax[i] * softmax[k]

    return J.tolist()