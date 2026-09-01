import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate descriptive statistics for a given dataset.

    Args:
        data: List or numpy array of numerical values

    Returns:
        Dictionary containing mean, median, mode, variance,
        standard deviation, percentiles, and IQR.
    """
    data = np.asarray(data)

    # Mean
    mean = np.mean(data)

    # Median
    median = np.median(data)

    # Mode
    values, counts = np.unique(data, return_counts=True)
    mode = values[np.argmax(counts)]

    # Variance and standard deviation
    variance = np.var(data)
    standard_deviation = np.std(data)

    # Percentiles
    percentile_25 = np.percentile(data, 25)
    percentile_50 = np.percentile(data, 50)
    percentile_75 = np.percentile(data, 75)

    # Interquartile range
    iqr = percentile_75 - percentile_25

    return {
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": variance,
        "standard_deviation": standard_deviation,
        "25th_percentile": percentile_25,
        "50th_percentile": percentile_50,
        "75th_percentile": percentile_75,
        "interquartile_range": iqr
    }