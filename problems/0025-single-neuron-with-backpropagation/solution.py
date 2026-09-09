import numpy as np

def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> tuple[np.ndarray, float, list[float]]:

    updated_weights = initial_weights.copy()
    updated_bias = initial_bias

    mse_values = []

    for _ in range(epochs):

        # -------------------------
        # 1. Forward pass
        # -------------------------
        z = np.dot(features, updated_weights) + updated_bias

        # Sigmoid activation
        y_hat = 1 / (1 + np.exp(-z))

        # -------------------------
        # 2. Calculate loss
        # -------------------------
        loss = np.mean((y_hat - labels) ** 2)
        mse_values.append(round(loss, 4))

        # -------------------------
        # 3. Backpropagation
        # -------------------------
        num_samples = len(labels)

        # dL / dy_hat
        d_loss_d_y_hat = 2 * (y_hat - labels) / num_samples

        # dy_hat / dz
        d_y_hat_d_z = y_hat * (1 - y_hat)

        # Chain rule:
        # dL/dz = dL/dy_hat * dy_hat/dz
        d_loss_d_z = d_loss_d_y_hat * d_y_hat_d_z

        # dL/dw
        d_loss_d_weights = np.dot(features.T, d_loss_d_z)

        # dL/db
        d_loss_d_bias = np.sum(d_loss_d_z)

        # -------------------------
        # 4. Gradient descent update
        # -------------------------
        updated_weights -= learning_rate * d_loss_d_weights
        updated_bias -= learning_rate * d_loss_d_bias

    return updated_weights, updated_bias, mse_values