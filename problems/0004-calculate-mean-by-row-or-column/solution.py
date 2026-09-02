import numpy as np
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	matrix = np.array((matrix))

	if mode == 'column':
	  return np.mean(matrix , axis = 0 )
	return np.mean(matrix , axis = 1 )
