import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    np.random.seed(seed)
    
    # 1. Determine population parameters and generate the required (runs, n) array
    if distribution == 'uniform':
        mu = 0.5
        sigma = np.sqrt(1 / 12)
        samples = np.random.uniform(0, 1, size=(runs, n))
        
    elif distribution == 'exponential':
        mu = 1.0
        sigma = 1.0
        samples = np.random.exponential(1.0, size=(runs, n))
        
    elif distribution == 'bernoulli':
        mu = 0.3
        sigma = np.sqrt(0.3 * 0.7)
        samples = (np.random.rand(runs, n) < 0.3).astype(float)
        
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
        
    # 2. Compute the sample means for each run (axis=1 collapses the n samples)
    sample_means = np.mean(samples, axis=1)
    
    # 3. Standardize the sample means to Z-scores using the Standard Error (sigma / sqrt(n))
    standard_error = sigma / np.sqrt(n)
    z_scores = (sample_means - mu) / standard_error
    
    # 4. Return the mean and population standard deviation (ddof=0) of the Z-scores
    return {
        'mean': float(np.mean(z_scores)),
        'std': float(np.std(z_scores, ddof=0))
    }
