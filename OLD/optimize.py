
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

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
    return amp * np.exp( - (x-x0)**2 / (2*sigma**2) )

def fit_gaussian(x,y,sigma=None,x0=None,amp=None,offset=None):
    '''
    1D Gaussian Fitting
    '''
    ### Guessing
    # average for the guess
    x_avg, y_avg = smooth.subsampleavg(x,y,bins=5)
    x_avg, y_avg = x.copy(), y.copy()
    # simple guess

def gaussian_2d(x, y, amp, cenx, ceny, sx, sy, angle, offset):
    '''
    inputs: x, y, amplitude, center_x, center_y, sigma_x, sigma_y, angle, offset
    output:
        x,y = x-cenx, y-ceny
        xp = x*np.cos(angle) - y*np.sin(angle)
        yp = x*np.sin(angle) + y*np.cos(angle)
        sxp, syp = 2*sx**2, 2*sy**2
        z = amp * np.exp(-xp**2/sxp - yp**2/syp) + offset
    '''
    return amp * np.exp(- ((x-cenx)*np.cos(angle) - (y-ceny)*np.sin(angle))**2/(2*sx**2) - ((x-cenx)*np.sin(angle) + (y-ceny)*np.cos(angle))**2/(2*sy**2) ) + offset

def surface_fit(*args, **kwargs):
    # Get kwargs
    fun = kwargs.get('fun',None)
    guess = kwargs.get('guess',None)
    disp = kwargs.get('disp',0)
    output = kwargs.get('output',False)
    show = kwargs.get('show',False)
    if fun is None or guess is None: raise TypeError("surface_fit: Must provide kwargs fun and guess")
    # Get x,y,z,using from inputs
    if len(args) is 0:
        return None
    elif len(args) <= 2:
        z = args[0]
        using = np.ones_like(z, dtype=np.bool) if len(args) is 1 else args[1].astype(np.bool)
        x, y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
    elif len(args) <= 4:
        x, y, z = args[0:3]
        using = np.ones_like(z, dtype=np.bool) if len(args) is 3 else args[3].astype(np.bool)
        if len(x.shape) is 1: x, y = np.meshgrid(x, y)
    else:
        raise ValueError

    # Prepare data for fitting
    using[np.logical_not(np.isfinite(z))] = False

    # fit
    def fitfunc(params):
        return np.ravel(z[using] - fun(x[using], y[using], *params))
    res = least_squares(fitfunc, guess, verbose=disp)

    # outputs
    res['zfit'] = fun(x, y, *res.x)

    # Show
    if show:
        fig,ax = plt.subplots(nrows=1,ncols=1)
        ax.imshow(z,cmap='gray')
        ax.contour(res['zfit'], cmap='jet')
        res['fig'] = fig

    if output: return res
    else: return res.x