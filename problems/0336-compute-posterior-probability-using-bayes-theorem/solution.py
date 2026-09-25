def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
    """
    Calculate posterior probabilities using Bayes' Theorem.
    """

    hypotheses = []

    for i in range(len(priors)):
        hypotheses.append(priors[i] * likelihoods[i])

    denominator = sum(hypotheses)

    result = [value / denominator for value in hypotheses]

    return result