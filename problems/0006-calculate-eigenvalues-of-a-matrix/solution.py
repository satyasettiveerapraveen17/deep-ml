import numpy as np
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	matrix = np.array(matrix)
	eigenvalues = np.linalg.eigvals(matrix ).real.tolist()
	return eigenvalues