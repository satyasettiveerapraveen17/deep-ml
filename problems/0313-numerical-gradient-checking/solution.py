import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):

    numerical_grad = np.zeros_like(x, dtype=float)

    for i in range(len(x)):

        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        f_plus = f(x_plus)
        f_minus = f(x_minus)

        numerical_grad[i] = (f_plus - f_minus) / (2 * epsilon)

    # Compare numerical and analytical gradients
    numerator = np.linalg.norm(numerical_grad - analytical_grad)
    denominator = (
        np.linalg.norm(numerical_grad)
        + np.linalg.norm(analytical_grad)
    )

    if denominator == 0:
        relative_error = 0.0
    else:
        relative_error = numerator / denominator

    return numerical_grad, relative_error