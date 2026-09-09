import numpy as np

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
    """
    Compute the derivative of cross-entropy loss with respect to logits.

    Args:
        logits: Raw model outputs (before softmax)
        target: Index of the true class (0-indexed)

    Returns:
        Gradient vector where gradient[i] = dL/d(logits[i])
    """
    logits = np.array(logits, dtype=float)

    # Softmax
    exp_x = np.exp(logits - np.max(logits))
    softmax = exp_x / np.sum(exp_x)

    # One-hot target
    y = np.zeros(len(logits))
    y[target] = 1.0

    # dL/d(logits) = softmax - target
    gradient = softmax - y

    return gradient.tolist()