import numpy as np
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	v= np.array(v)
	L= np.array(L)

	norms_L =np.sum(L**2)

	project = (v @ L / norms_L) * L
	return project 
