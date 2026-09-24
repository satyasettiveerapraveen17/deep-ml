import numpy as np 
def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
    return 0 if np.isnan(result.mean()) else round(result.mean(), 4)  x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    # Your code here
    data = np.array(data)
    condition = data[:, 0] == x
    result = data[condition, 1] == y
    return 0 if np.isnan(result.mean()) else round(result.mean(), 4) 