# Smoothening

# Imports
import numpy as np
import matplotlib.pyplot as plt


# Bin y1, y2, ... by x using given equally spaced bins array center
def binbyx_array_equal(*args,**kwargs):
    # Process inputs
    binsarray = kwargs.get('bins',None)
    emptybins = kwargs.get('emptybins',np.nan)
    binned = [None] * len(args)
    for i,dataarray in enumerate(binned): binned[i] = np.zeros(binsarray.shape)
    # Prepare variables
    delta = ( binsarray[1] - binsarray[0] )/2
    x = args[0]
    # Bin the arrays
    for i,bincenter in enumerate(binsarray):
        # Get the indices for i^th bin
        inds = np.logical_and( x >= (bincenter-delta) , x < (bincenter+delta) )
        # Bin all dataarrays for the i^th bin
        for j,dataarray in enumerate(args):
            members = dataarray[inds]
            if len(members) is 0: binned[j][i] = emptybins
            else: binned[j][i] = np.mean(members)
    # return the data
    return binned



def subsampleavg(*args,**kwargs):
	# Process inputs
	binLength = kwargs.get('bins',2)
	# Deal with exceptions
	if binLength <= 1: return args
	elif binLength >= len(args[0]): return [np.array(np.mean(data)) for data in args]
	# Calculate useful variables
	allowableLength = int(len(args[0])/binLength) * binLength
	binArrayLength = int(len(args[0])/binLength)
	# Prepare output variables
	binned = [None] * len(args)
	# Bin points upto allowableLength
	for i,data in enumerate(args):
		binned[i] = np.mean(np.reshape(data[:allowableLength],(binArrayLength,binLength)),axis=1)
	# Bin the leftover is there are any
	if allowableLength != len(args[0]):
		for i,data in enumerate(args):
			binned[i] = np.append(binned[i], np.mean(data[allowableLength:]))
	# Return
	return binned


def main():
	# Test of binbyx_array_equal

	# Define some data and bins
	x = np.linspace(0,3,150) ** 2
	y = np.sin(x)
	yn = y + np.random.normal(scale=0.2,size=x.shape)
	bins = np.linspace(0,9,30)
	# Binning the data with constant spacing for bins
	binsx,binsy = binbyx_array_equal(x,y,bins=bins)
	# Plot the results
	fig,axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
	axes.plot(x,y,'b-')
	axes.plot(x,yn,'r.')
	axes.plot(binsx, binsy,'go')
	plt.show()

	# Test of subsampleavg
	x = np.linspace(0,3,100)
	y = np.sin(x**2)
	yn = y + np.random.normal(scale=0.2,size=x.shape)
	binsx, binsy = subsampleavg(x,yn,bins=10)
	plt.plot(x,y,'b-',x,yn,'r.',binsx,binsy,'go')
	plt.show()



if __name__ == '__main__':
	main()



