import numpy as np

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
    """

    sigmoid = 1 / (1 + np.exp(-x))

    return {
        'sigmoid': float(sigmoid * (1 - sigmoid)),
        'tanh': float(1.0 - np.tanh(x)**2),
        'relu': 1.0 if x > 0 else 0.0
    }