# Smoothening

# Imports
import numpy as np
import matplotlib.pyplot as plt


### High level functions

# Binning data by x
def binbyx(*args, **kwargs):
    '''
    Binning data by x : binbyx(*args, **kwargs)

    Inputs: first pass the data (as many arrays as you want) x,y1,y2,...
            then keyword arguments to specify what kind of bin to use (details below)
    Output: list of binned data of same length as args [x_binned, y1_binned, ...]
    Keys:
        bins: number of bins, make 100 bins between x_min and x_max
        step (or steps): delta_x
        sects (or edges or edge or breaks or sect): defines sections
        blank (or blanks) : filler for blank bins (defualt np.nan)
        func : function handle, what to do with elements in a bin? (default np.mean)
        center_x : True returns the center of the x bins rather than the mean of the bin

    Examples:
        [x_binned, y1_binned, y2_binned] = binbyx(x,y1,y2,step=0.5)
        binbyx(x,y1,y2,bins=100)    or    binbyx(x,y,step=0.5)

    Possible kwargs:
        bins = 100    or    step = 0.1
        sects=50,bins=[25,25]    or    sects=[10,80,120],step=[0.1,1]

    Understanding sections:
        binbyx(x,y,sects=[2,10,21,26],bins=[3,5,2])
        x = 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
              __________________ ________________________________ ______________
                 3 bins here            5 bins in this section      2 bins here

    Procedure:

    '''
    # Process *args
    x = args[0]
    x_min,x_max = np.min(x),np.max(x)
    # Default values
    binsdef = [None]
    stepdef = [None]
    blankdef = np.nan
    sectsdef = [x_min,x_max]
    # Load inputs
    bins = kwargs.get('bins',binsdef)
    if type(bins) is tuple: bins = list(bins)
    step = kwargs.get('step',None)
    if step == None: step = kwargs.get('steps',stepdef)
    if type(step) is tuple: step = list(step)
    blank = kwargs.get('blank',None)
    if blank == None: blank = kwargs.get('blanks',blankdef)
    sects = kwargs.get('sects',None)
    if sects == None: sects = kwargs.get('edges',None)
    if sects == None: sects = kwargs.get('edge',None)
    if sects == None: sects = kwargs.get('breaks',None)
    if sects == None: sects = kwargs.get('sect',sectsdef)
    if type(sects) is tuple: sects = list(sects)
    func = kwargs.get('func',np.mean)
    # Make it a list if it is not already
    if type(bins) != list: bins = [bins]
    if type(sects) != list: sects = [sects]
    if type(step) != list: step = [step]
    # Pad sects with x_min, x_max if not provided
    if len(sects) == max(len(bins),len(step))-1: sects = [x_min,*sects,x_max]
    if len(sects) != max(len(bins),len(step))+1: print('Input Error: Place discription here!')
    # Prepare outputs
    binsarray = np.array([])
    # Compute bins array
    if bins[0] != None: # bins are provided
        for i,ibins in enumerate(bins):
            binsarray = np.append( binsarray, np.linspace(sects[i],sects[i+1],bins[i]+1)[0:-1] )
    elif step[0] != None: # steps are provided
        for i,istep in enumerate(step):
            binsarray = np.append( binsarray, np.arange(sects[i],sects[i+1],step[i]) )
    else: # nothing was provided, default case
        binsarray = np.linspace(sects[0],sects[1],11)[0:-1]
    # Bin and return the data
    binsarray = binsarray + 0.5*(binsarray[1]-binsarray[0])
    binned = binbyx_array(*args,bins=binsarray,emptybins=blank,func=func)
    if kwargs.get('center_x', False):
        binned[0] = binsarray
    return binned


### Low level functions

# Bin y1, y2, ... by x using given equally spaced bins array center
def binbyx_array_equal(*args,**kwargs):
    # Warming
    print('please REPLACE binbyx_array_equal WITH binbyx')
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
            if len(members) == 0: binned[j][i] = emptybins
            else: 
                if j==0: binned[j][i] = np.mean(members)
                else: binned[j][i] = np.mean(members)
    # return the data
    return binned

# Bin y1, y2, ... by x using given bins array center
def binbyx_array(*args,**kwargs):
    # Process inputs
    binsarray = kwargs.get('bins',None)
    emptybins = kwargs.get('emptybins',np.nan)
    func = kwargs.get('func',np.mean)
    binned = [None] * len(args)
    for i,dataarray in enumerate(binned): binned[i] = np.zeros_like(binsarray)
    # Prepare variables
    N = binsarray.size
    delta = np.diff(binsarray) / 2.0
    edges = np.zeros(N+1)
    # Define edges
    x = args[0]
    edges[0] = binsarray[0] - delta[0]
    edges[N] = binsarray[N-1] + delta[N-2]
    for i in range(1,N): edges[i] = binsarray[i-1] + delta[i-1]
    # Bin the array
    for i,bincenter in enumerate(binsarray):
        # Get the members for i^th bin
        inds = np.logical_and( x >=  edges[i], x < edges[i+1] )
        # Bin all dataarrays for the i^th bin
        for j,dataarray in enumerate(args):
            members = dataarray[inds]
            if len(members) == 0: binned[j][i] = emptybins
            else:
                if j == 0: binned[j][i] = np.nanmean(members)
                else: binned[j][i] = func(members)
    # Return the data
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


def subsample2D(x, bins=(1,1)):
    '''
        Rebin a 2D array into a smaller 2D array. Pixels within bins[0] x bins[1] are summed up into the returned array.
        Make sure there are no inf and nan in the 2D array.
    '''
    # Error Checking
    if len(x.shape) != 2 and len(bins) != 2:
        raise ValueError("subsample2D is ONLY for 2d array and bins must be of type (2,2).")
    if bins[0] > x.shape[0] or bins[1] > x.shape[1]:
        print("WARNING! subsample2D received bins larger than the shape of 2D array. bins will be set to shape of x")
        bins = x.shape
    # Procedure
    finalSize = np.array(np.asarray(x.shape) / np.asarray(bins), dtype=np.int)
    initialSize = finalSize * bins
    return x[0:initialSize[0], 0:initialSize[1]].reshape(finalSize[0], bins[0], finalSize[1], bins[1]).sum(1).sum(2)


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


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



