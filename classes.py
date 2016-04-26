# Define classes

# Imports
import numpy as np
import matplotlib.pyplot as plt


# Importing from therpy is its main else using .
if __name__ == '__main__':
	import os.path
	import sys
	path2therpy = os.path.join(os.path.expanduser('~'), 'Documents', 'My Programs', 'Python Library')
	sys.path.append(path2therpy)
	from therpy import smooth
	from therpy import calculus
	from therpy import imageio
	from therpy import imagedata
	from therpy import constants
else:
	from . import smooth
	from . import calculus
	from . import imageio
	from . import imagedata
	from . import constants

#import smooth
#import calculus

# Curve class for 1D functions
class Curve:
	'''
	class for generic function f(xi) = yi
	'''

	def __init__(self, x=np.array([]), y=np.array([])):
		self.x = x.copy()
		self.y = y.copy()
		self.xLatest = x.copy()
		self.yLatest = y.copy()

	def __call__(self,xi,method='nearest'):
		# Curve(x) returns yi for the i such that x ~ xi
		pass 

	def subsample(self, bins=2):
		self.xSubSample, self.ySubSample = smooth.subsampleavg(self.x, self.y, bins=bins)
		self.xLatest = self.xSubSample.copy()
		self.yLatest = self.ySubSample.copy()
		return (self.xLatest, self.yLatest)

	def binbyx_ae(self, bins):
		self.xBinByX, self.yBinByX = smooth.binbyx_array_equal(self.x, self.y, bins=bins)
		self.xLatest = self.xBinByX.copy()
		self.yLatest = self.yBinByX.copy()
		return (self.xLatest, self.yLatest)

	def diff(self, **kwargs):
		method = kwargs.get('method','poly')
		self.xDiff = self.xLatest
		self.yDiff = self.yLatest
		if method=='poly':
			self.dydx = calculus.numder_poly(self.xDiff, self.yDiff, order=kwargs.get('order',1), points=kwargs.get('points',2))
		elif method == 'gaussian filter':
			self.dydx = calculus.numder_gaussian_filter(self.xDiff, self.yDiff, order=1, sigma=kwargs.get('sigma',1))
		return (self.xDiff, self.dydx)

	def int(self, **kwargs):
		method = kwargs.get('method','sum')
		self.xInt = self.xLatest
		self.yInt = self.yLatest
		if method=='sum':
			self.Int = np.sum(self.y) * self.dx
		return self.Int

	@property
	def dx(self):
	    return (self.x[1] - self.x[0])
	
	@property
	def plotdata(self):
	    return (self.xLatest, self.yLatest)


# Absorption Image Class
class AbsImage():
	'''
	Absorption Image Class
	Inputs:
	name (image filename with date prefix) or
	wa and woa (numpy arrays)
	cst object
	'''
	def __init__(self, **kwargs):
		# Create a dict var to store all information
		self.var = kwargs
		self.cst = kwargs.get('cst',constants.cst())

	# Universal Properties
	@property
	def wa(self):
		if 'wa' not in self.var.keys():
			alldata = self.alldata
			self.var['wa'] = alldata[0]
			self.var['woa'] = alldata[1]
		return self.var.get('wa')
	
	@property
	def woa(self):
		if 'woa' not in self.var.keys():
			alldata = self.alldata
			self.var['wa'] = alldata[0]
			self.var['woa'] = alldata[1]
		return self.var.get('woa')

	@property
	def od(self):
		with np.errstate(divide='ignore',invalid='ignore'):
			return np.log(self.woa/self.wa)

	@property
	def ncol(self): 
		part1 = self.od * self.cst.ODf
		part2 = (self.woa-self.wa)/self.cst.Nsat
		return (part1 + part2) / self.cst.sigma

	@property
	def atoms(self):
		return self.ncol * self.cst.pixel**2

	@property
	def total_atoms(self):
		return np.sum(self.atoms)


	
	# Properties special for fits images
	@property
	def name(self): return self.var.get('name','NotGiven')

	@property
	def path(self): return imageio.imagename2imagepath(self.name)
	
	@property
	def rawdata(self): return imageio.imagepath2imagedataraw(self.path)

	@property
	def alldata(self): return imageio.imagedataraw2imagedataall(self.rawdata)
	
	
	# Crop index function
	def cropi(self,**kwargs): return imagedata.get_cropi(self.od,**kwargs)



	# Useful functions
	def __str__(self):
		des = 'Absorption Image: ' + self.name + ' Image size: ' + str(self.od.shape)
		return des
	



# Main
def main():
	# # Tests of Curve
	# xi = np.linspace(0,3,100)
	# yi = np.sin(xi**2)
	# dydx = np.cos(xi**2) * 2 * xi
	# yin = yi + np.random.normal(scale=0.1,size=xi.shape)
	# curve0 = Curve(xi,yi)
	# curve1 = Curve(xi, yin)

	# plt.plot(*curve0.plotdata,'b-')
	# plt.plot(*curve1.subsample(bins=4),'r.')
	# plt.plot(xi, dydx,'b-')
	# plt.plot(*curve1.diff(method='poly'),'ko')
	# plt.plot(*curve1.diff(method='gaussian filter'),'gs')
	# plt.show()

	# print(curve0.int())
	# print(curve0.dx)

	wa = np.ones((512,512));
	woa = np.ones_like(wa) + 0.1;

	img1 = AbsImage(wa=wa,woa=woa)
	print(img1.name)
	img2 = AbsImage(name='03-24-2016_21_04_12_top')
	print(img2.rawdata.shape)
	print(img2)

if __name__ == '__main__':
	main()


