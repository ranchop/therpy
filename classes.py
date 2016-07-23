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


# import smooth
# import calculus

# Curve class for 1D functions
class Curve:
    """
    class for generic function f(xi) = yi

    Properties:
        x, y, data, dx, sorti

    Methods:
        __call__
        inverse (in progress)
        loc (in progress)
        sortbyx : returns sorted (x,y)
        binbyx : returns binned (x,y)
        subsample : returns sub sampled (x,y)
        diff (in progress)
        int (in progress)
    """

    def __init__(self, x=None, y=np.array([])):
        if x is None: x = np.arange(y.size)
        self.var = {'x': x.copy(), 'y': y.copy()}

    ### Properties ###

    @property
    def x(self):
        return self.var.get('x', np.array([]))

    @property
    def y(self):
        return self.var.get('y', np.array([]))

    @property
    def yfit(self):
        return self.var.get('yfit', None)

    @property
    def xyfitplot(self):
        return self.var.get('xyfitplot', None)

    @property
    def sorti(self):
        sorti = self.var.get('sorti', None)
        if sorti is None:
            sorti = np.argsort(self.x)
            self.var['sorti'] = sorti
        return sorti

    @property
    def dx(self):
        return (self.x[1] - self.x[0])

    @property
    def data(self):
        return (self.x, self.y)

    @property
    def max(self):
        return (None)

    ### High level methods ###
    def __call__(self, xi):
        return self.y[self.sorti][self.locx(xi)]

    def __str__(self):
        des = 'A curve with ' + str(self.x.size) + ' data points.'
        return des

    def inverse(self, yi):
        pass

    def loc(self, x=None, y=None):
        if x != None:
            return self.locx(x)
        elif y != None:
            return self.locy(y)
        else:
            print('ERROR: Please provide x or y')
        return 0

    def sortbyx(self):
        return (self.x[self.sorti], self.y[self.sorti])

    def binbyx(self, **kwargs):
        return smooth.binbyx(self.x, self.y, **kwargs)

    def subsample(self, bins=2):
        return smooth.subsampleavg(self.x, self.y, bins=bins)

    def diff(self, **kwargs):
        method = kwargs.get('method', 'poly')
        if method == 'poly':
            dydx = calculus.numder_poly(self.x, self.y, order=kwargs.get('order', 1), points=kwargs.get('points', 1))
        elif method == 'central2':
            dydx = np.gradient(self.y, self.dx, edge_order=2)
        return (self.x, dydx)

    def fit(self, fitfun, guess, plot=False, pts=1000):
        # Fit
        from scipy.optimize import curve_fit
        fitres, fiterr = curve_fit(fitfun, self.x, self.y, p0=guess)
        fiterr = np.sqrt(np.diag(fiterr))
        yfit = fitfun(self.x, *fitres)
        xfitplot = np.linspace(np.min(self.x), np.max(self.x), pts)
        yfitplot = fitfun(xfitplot, *fitres)
        # Save results in var
        self.var['yfit'] = yfit
        self.var['xyfitplot'] = (xfitplot, yfitplot)
        self.var['fitres'] = fitres
        self.var['fiterr'] = fiterr
        # Plot and display
        if plot:
            # Plot
            plt.figure(figsize=(5, 7))
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0))
            ax1.plot(*self.xyfitplot,'k-',*self.data,'r.')
            ax2.plot(self.x, self.y-self.yfit,'r.')
            ax2.vlines(self.x, self.x*0, self.y-self.yfit)
            plt.xlabel('x')
            plt.ylabel('Difference')
            # Print
            print("##______Fit Value______Error______")
            for i,val in enumerate(fitres):
                print("{:2d} ==> {:9.4} (+-) {:9.4}".format(i, fitres[i], fiterr[i]))
        # return fitresults
        return (fitres, fiterr)

    ### Low level methods ###
    def locx(self, xi):
        x = self.x[self.sorti]
        iloc = np.argwhere(x <= xi)
        if len(iloc) == 0:
            return 0
        elif len(iloc) == x.size:
            return x.size - 1
        else:
            iloc = iloc[-1, 0]
        if (xi - x[iloc]) >= (x[iloc + 1] - xi): iloc += 1
        return iloc

    def locy(self, yi):
        pass

    def int(self, **kwargs):
        method = kwargs.get('method', 'sum')
        self.xInt = self.xLatest
        self.yInt = self.yLatest
        if method == 'sum':
            self.Int = np.sum(self.y) * self.dx
        return self.Int


# Absorption Image Class
class AbsImage():
    '''
    Absorption Image Class

    Inputs:
        one of the three ways to define an image
        constants object (default is Li 6 Top Imaging)


    Inputs (Three ways to define an image):
        1) name (image filename with date prefix)
        2) wa and woa (numpy arrays)
        3) od (numpy array)
    cst object
    '''

    def __init__(self, *args, **kwargs):
        # Create a dict var to store all information
        self.var = kwargs
        self.cst = kwargs.get('cst', constants.cst())
        # Check the args
        if len(args) > 0 and type(args[0]) is str: self.var['name'] = args[0]

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
        if 'od' not in self.var.keys():
            self.var['od'] = imagedata.get_od(self.wa, self.woa, width=self.var.get('trouble_pts_width', 5))
        return self.var['od']

    @property
    def rawod(self):
        if 'rawod' not in self.var.keys():
            self.var['rawod'] = imagedata.get_od(self.wa, self.woa, width=self.var.get('trouble_pts_width', 5),
                                                 rawod=True)
        return self.var['rawod']

    @property
    def ncol(self):
        part1 = self.od * self.cst.ODf
        part2 = (self.woa - self.wa) / self.cst.Nsat
        return (part1 + part2) / self.cst.sigma

    @property
    def atoms(self):
        return self.ncol * self.cst.pixel ** 2

    @property
    def total_atoms(self):
        return np.sum(self.atoms)

    @property
    def xy(self):
        x = np.arange(self.wa.shape[1])
        y = np.arange(self.wa.shape[0])
        return np.meshgrid(x, y)

    # Properties special for fits images
    @property
    def name(self):
        return self.var.get('name', 'NotGiven')

    @property
    def path(self):
        return imageio.imagename2imagepath(self.name)

    @property
    def rawdata(self):
        return imageio.imagepath2imagedataraw(self.path)

    @property
    def alldata(self):
        return imageio.imagedataraw2imagedataall(self.rawdata)

    # Crop index function
    def cropi(self, **kwargs):
        return imagedata.get_cropi(self.od, **kwargs)

    # fix intensities
    def fixVaryingIntensities_AllOutside(self, xmin, xmax, ymin, ymax):
        # Define a crop region and find factor*woa
        (x,y) = self.xy
        cropi = np.logical_and.reduce((x>=xmin, x<=xmax, y>=ymin, y<=ymax))
        factor = np.sum(self.alldata[0][cropi==0]) / np.sum(self.alldata[1][cropi==0])
        self.var['factor_woa'] = factor
        # Recalculate wa, woa, od, rawod
        self.var['wa'] = self.alldata[0]
        self.var['woa'] = self.alldata[1] * self.var['factor_woa']
        self.var['od'] = imagedata.get_od(self.var['wa'], self.var['woa'], width=self.var.get('trouble_pts_width', 5))
        self.var['rawod'] = imagedata.get_od(self.wa, self.woa, width=self.var.get('trouble_pts_width', 5), rawod=True)

    def fixVaryingIntensities_Box(self, cropi=None, **kwargs):
        # Define a crop region and find factor*woa
        (x,y) = self.xy
        if cropi is None: cropi = self.cropi(**kwargs)
        factor = np.sum(self.alldata[0][cropi]) / np.sum(self.alldata[1][cropi])
        self.var['factor_woa'] = factor
        # Recalculate wa, woa, od, rawod
        self.var['wa'] = self.alldata[0]
        self.var['woa'] = self.alldata[1] * self.var['factor_woa']
        self.var['od'] = imagedata.get_od(self.var['wa'], self.var['woa'], width=self.var.get('trouble_pts_width', 5))
        self.var['rawod'] = imagedata.get_od(self.wa, self.woa, width=self.var.get('trouble_pts_width', 5), rawod=True)

    # Averaging multiple images together
    def avgod(self, *args):
        avg = self.od
        for im in args: avg += im.od
        return avg / (1 + len(args))

    # pixels that are not usable are defined by:
    def usable(self, threshold=25):
        return imagedata.get_usable_pixels(self.wa, self.woa, threshold=threshold)

    # Fixing nan and inf
    # Example
    # indices = np.where(np.isnan(a)) #returns an array of rows and column indices
    # for row, col in zip(*indices):
    # a[row,col] = np.mean(a[~np.isnan(a[:,col]), col]) need to modify this

    # def interpolate_nans(X):
    # """Overwrite NaNs with column value interpolations."""
    # for j in range(X.shape[1]):
    # 	mask_j = np.isnan(X[:,j])
    # 	X[mask_j,j] = np.interp(np.flatnonzero(mask_j), np.flatnonzero(~mask_j), X[~mask_j,j])
    # return X


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

    wa = np.ones((512, 512));
    woa = np.ones_like(wa) + 0.1;

    img1 = AbsImage(wa=wa, woa=woa)
    print(img1.name)
    img2 = AbsImage(name='03-24-2016_21_04_12_top')
    print(img2.rawdata.shape)
    print(img2)


if __name__ == '__main__':
    main()
