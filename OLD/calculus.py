# Calculus: derivative and integrals

# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.fftpack


# Derivatives
def numder_poly(x,y,order=2,points=1):
    # Prepare outputs
    dydx = np.zeros_like(y)
    # Useful variables
    start_index = points
    end_index = len(x) - points # not including
    # Bulk points
    for i in range(start_index, end_index):
        fitx = x[i-points:i+points+1]
        fity = y[i-points:i+points+1]
        fitpoly = np.poly1d(np.polyfit(fitx,fity,order))
        fitpolyder = np.polyder(fitpoly)        
        dydx[i] = fitpolyder(x[i])
    # Edge points
    for i in range(0,start_index): # Begging points
        dydx[i] = np.nan    
    for i in range(end_index,len(x)): # End points
        dydx[i] = np.nan    
    # Return results
    return dydx


# IN WORKS
def numder_gaussian_filter(x,y,**kwargs):
    # Process inputs
    sigma = kwargs.get('sigma',2)
    mode = kwargs.get('mode','constant')
    order = kwargs.get('order',1)
    # Useful variables
    dx = x[1]- x[0]
    # Differentiate
    dydx = scipy.ndimage.filters.gaussian_filter(y, sigma = sigma, mode=mode, order=order) / dx
    # Return results
    return dydx


def main():
    ## Test of numder_poly
    # Generate data and its derivative
    x = np.linspace(-3,3,200)
    y1 = np.sin(x**2) - np.log(np.abs(x))
    dy1 = np.gradient(y1,(x[1]-x[0]))
    # Add noise and generate polynomial derivative
    y1n = y1 + np.random.normal(scale = 0.5, size = x.shape)
    dy1n = numder_poly(x,y1n,order=2,points=4) 
    dy2n = numder_gaussian_filter(x,y1n,sigma=3)
    # Plot results
    fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axes[0].plot(x,y1,'r-')
    axes[0].plot(x,y1n,'b.')
    axes[1].plot(x,dy1,'r-')
    axes[1].plot(x,dy1n,'b.')
    axes[1].plot(x,dy2n,'g-')
    axes[1].set_ylim([-15,15])
    plt.show()

if __name__ == '__main__':
	main()


