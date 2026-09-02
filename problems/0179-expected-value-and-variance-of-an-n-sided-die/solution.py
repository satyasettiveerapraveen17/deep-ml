def dice_statistics(n: int) -> tuple[float, float]:
	"""
	Compute the expected value and variance of a fair n-sided die roll.

	Args:
		n (int): Number of sides of the die

	Returns:
		tuple: (expected_value, variance)
	"""
    # NOTE : E(x) = n+1 / 2
	# Var(x) = Square of E(x) - E(square of x ) = n sqaure - 1 / 12 


	return ( (n+1)/2  , (n**2 - 1) / 12)
	