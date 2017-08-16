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
    step = kwargs.get('step',None)
    if step == None: step = kwargs.get('steps',stepdef)
    blank = kwargs.get('blank',None)
    if blank == None: blank = kwargs.get('blanks',blankdef)
    sects = kwargs.get('sects',None)
    if sects == None: sects = kwargs.get('edges',None)
    if sects == None: sects = kwargs.get('edge',None)
    if sects == None: sects = kwargs.get('breaks',None)
    if sects == None: sects = kwargs.get('sect',sectsdef)
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
    return binbyx_array(*args,bins=binsarray,emptybins=blank,func=func)


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



