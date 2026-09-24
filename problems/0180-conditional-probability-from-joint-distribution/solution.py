import numpy as np

def conditional_probability(joint_distribution: dict) -> float:
    """
    Calculate P(A | B) from a joint probability distribution.
    """

    p_ab = joint_distribution[("A", "B")]

    # P(B) = P(A,B) + P(¬A,B)
    p_b = (
        joint_distribution[("A", "B")]
        + joint_distribution[("`A", "B")]
    )

    return 0 if p_b == 0 else round(p_ab / p_b, 4)