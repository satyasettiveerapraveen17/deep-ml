import numpy as np
def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    values, counts = np.unique(np.array(samples), return_counts=True)
    
    pmf = []
    
    for value, count in zip(values, counts):
        result = (value, count / len(samples))
        pmf.append(result)
    return pmf
    