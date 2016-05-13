
import numpy as np
from scipy.optimize import curve_fit

# Importing from therpy is its main else using .
if __name__ == '__main__':
	import os.path
	import sys
	path2therpy = os.path.join(os.path.expanduser('~'), 'Documents', 'My Programs', 'Python Library')
	sys.path.append(path2therpy)
	from therpy import smooth
else:
	from . import smooth

# Optimization : curve and surface fit
# A good overview of different optimization in scipy.optimize is in notebook

### specific functions fitting
def fun_gaussian(x,sigma=1,x0=0,amp=1,offset=0):
	'''
	p1 = (x-x0)**2
	p2 = 2 * sigma**2
	amp * np.exp(p1/p2)
	'''
	return amp * np.exp( (x-x0)**2 / (2*sigma**2) )

def fit_gaussian(x,y,sigma=None,x0=None,amp=None,offset=None):
	'''
	1D Gaussian Fitting
	'''
	### Guessing
	# average for the guess
	x_avg, y_avg = smooth.subsampleavg(x,y,bins=5)
	x_avg, y_avg = x.copy(), y.copy()
	# simple guess
	