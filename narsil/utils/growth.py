# Functions that help with growth rates
import numpy as np

def exp_growth_fit(x, a, b):
	return a * np.exp(-b * x)

