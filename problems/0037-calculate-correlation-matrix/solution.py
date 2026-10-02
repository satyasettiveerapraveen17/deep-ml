import numpy as np

def calculate_correlation_matrix(X, Y=None):
    X = np.array(X)
    
    if Y is None:
        # Standard correlation matrix of X columns against themselves
        return np.corrcoef(X, rowvar=False)
    else:
        Y = np.array(Y)
        # 1. Get the full joint correlation matrix
        full_matrix = np.corrcoef(X, Y, rowvar=False)
        
        # 2. Find how many features (columns) are in X
        n_features_X = X.shape[1] if X.ndim > 1 else 1
        
        # 3. Slice the top-right quadrant (X vs Y correlations)
        return full_matrix[:n_features_X, n_features_X:]

