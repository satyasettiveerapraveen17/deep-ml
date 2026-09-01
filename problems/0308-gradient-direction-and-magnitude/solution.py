import torch

def gradient_direction_magnitude(gradient: list) -> dict:
    """
    Calculate the magnitude and direction of a gradient vector.
    """
    gradient = torch.tensor(gradient, dtype=torch.float32)

    magnitude = torch.linalg.vector_norm(gradient, ord=2)

    if magnitude.item() == 0:
        return {
            "magnitude": 0.0,
            "direction": torch.zeros_like(gradient),
            "descent_direction": torch.zeros_like(gradient)
        }

    direction = gradient / magnitude
    descent_direction = -direction

    return {
        "magnitude": magnitude.item(),
        "direction": direction,
        "descent_direction": descent_direction
    }