# Define classes

# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spintp
import os
import os.path
import urllib.request
import numba
import pickle
import scipy.misc
import scipy.optimize
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    from therpy import functions
    from therpy import roots1
else:
    from . import smooth
    from . import calculus
    from . import imageio
    from . import imagedata
    from . import constants
    from . import functions
    from . import roots1


# import smooth
# import calculus


###################################################################################
############################ interp_od ############################################
# Load pre-compiled data
p_ = roots1.getpath('Projects','Data','LookupTable','Lookup_Table_Fast_PreCompData_V2.p')
if not os.path.isfile(p_):
    print("Downloading Database -- Might take some time!")
    url = "https://www.dropbox.com/s/4hklnvawtshjay9/Lookup_Table_Fast_PreCompData_V2.p?dl=1"
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    # Create folder
    os.makedirs(os.path.split(p_)[0], exist_ok=True)
    with open(p_, "wb") as f :
        f.write(data)
precompiled_data_Lookup_Table = pickle.load( open( p_, "rb" ) )

# Jitted interpolation
@numba.jit(nopython=True)
def interp_od_special_jit(IivIn, IfvIn, sim_data):
    # Unload sim_data
    u_si, sf_2d, ocd_2d = sim_data[0], sim_data[1], sim_data[2]
    rows, cols = sf_2d.shape[0], sf_2d.shape[1]
    # Copy inputs and flatten the arrays
    Iiv = IivIn.copy().flatten()  # Flatten so that we can do 1d loop
    Ifv = IfvIn.copy().flatten()  # We will unflatten the arrays when returning
    # Fix low and high OD regions
    bad_low = (Iiv < Ifv)  # For low od (BG), flip Ii and If and make od -> -od
    Iiv[bad_low], Ifv[bad_low] = Ifv[bad_low].copy(), Iiv[bad_low].copy()
    bad_high = (Ifv < 0)   # For high od where If < 0, make If -> -If
    Ifv = np.abs(Ifv)
    # Prepare
    i0v = np.searchsorted(u_si, Iiv)   # Find the indice for closest si
    Pfv = np.zeros_like(Iiv) * np.nan  # Prepare output array, default it with nan
    # Interpolate OD's
    for i in range(Iiv.size):
        Ii, If, i0 = Iiv[i], Ifv[i], i0v[i]
        # Search 4 closest points
        if i0 >= rows or i0 == 0: continue  # If Ii is outside simulation, result is nan
        i1 = np.searchsorted(sf_2d[i0-1,:], If)
        if i1 >= cols: Pfv[i] = 0; continue # If If > max(sf), result is zero atoms
        elif i1 == 0: continue
        i2 = np.searchsorted(sf_2d[i0,:], If)
        if i2 >= cols: Pfv[i] = 0; continue # If If > max(sf), result is zero atoms
        elif i2 == 0: continue
        i0m1 = i0-1
        x1 = u_si[i0m1]
        x2 = u_si[i0]
        dx = x2 - x1
        dx2 = dx**2
        Ary = sf_2d[i0m1, i1-1]
        Bry = sf_2d[i0, i2-1]
        Cry = sf_2d[i0m1, i1]
        Dry = sf_2d[i0, i2]
        Af = ocd_2d[i0m1, i1-1]
        Bf = ocd_2d[i0, i2-1]
        Cf = ocd_2d[i0m1, i1]
        Df = ocd_2d[i0, i2]
        # Interpolate with 4 nearest points
        s = (Ii - x1) / (dx)
        Erx = x1 + (dx) * s
        Ery = Ary + (Bry - Ary) * s
        Frx = x1 + (dx) * s
        Fry = Cry + (Dry - Cry) * s
        Ef = Af + (Bf - Af)  * (((Erx - x1)**2 + (Ery - Ary)**2) / ((dx2 + (Bry - Ary)**2)))**0.5
        Ff = Cf + (Df - Cf)  * (((Frx - x1)**2 + (Fry - Cry)**2) / ((dx2 + (Dry - Cry)**2)))**0.5
        Pfv[i] = Ef + (Ff - Ef) * (((Ii - Erx)**2 + (If - Ery)**2) / (((Frx - Erx)**2 + (Fry - Ery)**2)) )**0.5
    # Make the bad_low od -> -od
    Pfv[bad_low] *= -1
    # Reshape and return
    return Pfv.reshape(*IivIn.shape)

# Wrapper around jitted function to handle passing in pre-compiled data
def interp_od(Ii, If, img_time):
    return interp_od_special_jit(Ii, If, precompiled_data_Lookup_Table[img_time-1])


####################################################################################
########################### Image ##################################################
# Complete Image Class

Default_Image_Set = dict(name='Not Provided', path='Not Provided',
                        center_x=1, center_y=1, width=1000000, height=1000000,
                        subsample=1, rotate=0, rotate_method='bilinear',
                        prep_order=['rotate','crop','subsample'],
                        fudge=1, bg_width=0, bg_order=1, bad_light=0,
                        Isat=1, time=1, pixel=1e-6, detuning=0,
                        od_method='log', sigmaf=1, memory_saver=False,
                        lookup_table_version='v1')

Level_Selector_Image = [['name','path','center_x','center_y','center',
                         'width','height','cropset','cropi','subsample',
                         'rotate','rotate_method','prep_order'],
                        ['bg_width','bg_order'],
                        ['bad_light','Isat','time','od_method']]

'''
Speed: total of 350 ms per image
- 200 ms to load image into memory
- 100 ms to crop and subsample image
- 600 ms if rotating an image
- 5 ms to find border gradient
- 20 ms to computer OD using LookupTable (depends on crop size)
'''

cst_Image_Class = constants.cst()

class Image:
    '''Get image name and path, start self.var'''
    def __init__(self, name=None, path=None, od=None, **kwargs):
        # local storage
        self.var = {**Default_Image_Set, **kwargs}
        self.var['Level_Selector'] = list(Level_Selector_Image)
        self.var['recalc'] = [True]*len(self.var['Level_Selector'])

        # Use path if provided, else use name and find path
        if (type(path) is str) and os.path.exists(path):
            self.var['path'], self.var['name'] = path, os.path.splitext(os.path.split(path)[1])[0]
        elif type(name) is str: self.var['name'], self.var['path'] = name, imageio.imagename2imagepath(name)
        elif od is not None:
            self.var['od'] = od
            self.var['Level_Selector'][0] = [] # Disable Level, 0 computations
            self.var['Level_Selector'][1] = [] # Disable Level, 1 computations
            self.var['Level_Selector'][2] = [] # Disable Level, 2 computations
            self.var['recalc'] = [False]*len(self.var['Level_Selector'])
        else: raise ValueError('Please provide at least name, path, or od to the Image constructor.')

##### Properties -- Atomic Density and Numbers [in progress]
    @property
    def od(self,):
        if ('od' not in self.var.keys()) or self.recalc[2]:
            self.optical_density()
        return self.var['od'] * self.fudge

    @property
    def n_2d(self,): return self.od / self.sigma

    @property
    def app(self,): return self.od / self.sigma * self.pixel_binned**2

    @property
    def od_raw(self,): return - np.log(self.If_raw / self.Ii_raw)

    @property
    def total_atoms(self,): return np.nansum(self.app)
#####

##### Properties -- Imaging Intensity [in progress]
    @property
    def rawdata(self,): return imageio.imagepath2imagedataraw(self.path)

    @property
    def alldata(self,):
        if 'alldata' in self.var.keys(): return self.var.get('alldata')
        alldata = imageio.imagedataraw2imagedataall(self.rawdata)
        if self.memory_saver is False: self.var['alldata'] = alldata
        return alldata

    @property
    def Ii_raw(self,):
        if ('Ii_raw' not in self.var.keys()) or self.recalc[0]:
            self.prep_image()
        return self.var['Ii_raw']

    @property
    def If_raw(self,):
        if ('If_raw' not in self.var.keys()) or self.recalc[0]:
            self.prep_image()
        return self.var['If_raw']

    @property
    def alpha_Ii(self,):
        if ('alpha_Ii' not in self.var.keys()) or self.recalc[1]:
            self.border_gradient()
        return self.var['alpha_Ii']

    @property
    def Ii(self,): return (self.Ii_raw * self.alpha_Ii) * (1-self.bad_light)

    @property
    def If(self,): return self.If_raw - (self.Ii_raw * self.alpha_Ii * self.bad_light)

    @property
    def si(self,): return self.Ii / self.Nsat

    @property
    def sf(self,): return self.If / self.Nsat

    @property
    def Ii_avg(self,): return np.nanmean(self.Ii) / self.subsample**2

    @property
    def Ii_avg_binned(self,): return np.nanmean(self.Ii)

    @property
    def si_avg(self,): return np.nanmean(self.si)

#####

##### Properties -- Settings [completed]
    @property
    def name(self,): return self.var.get('name')

    @property
    def path(self,): return self.var.get('path')

    @property
    def center_x(self,): return self.var.get('center_x')

    @property
    def center_y(self,): return self.var.get('center_y')

    @property
    def center(self,): return self.var.get('center', (self.center_x, self.center_y))

    @property
    def width(self,): return self.var.get('width')

    @property
    def height(self,): return self.var.get('height')

    @property
    def cropset(self,): return self.var.get('cropset', dict(center=self.center, width=self.width, height=self.height))

    @property
    def cropi(self,):
        if ('cropi' not in self.var.keys()) or self.recalc[0]:
            self.prep_image()
        return self.var['cropi']

    @property
    def subsample(self,): return self.var.get('subsample')

    @property
    def rotate(self,): return self.var.get('rotate')

    @property
    def rotate_method(self,): return self.var.get('rotate_method')

    @property
    def prep_order(self,): return self.var.get('prep_order')

    @property
    def fudge(self,): return self.var.get('fudge')

    @property
    def bg_width(self,): return self.var.get('bg_width')

    @property
    def bg_order(self,): return self.var.get('bg_order')

    @property
    def bad_light(self,): return self.var.get('bad_light')

    @property
    def Isat(self,): return self.var.get('Isat')

    @property
    def Nsat(self,): return self.Isat * self.time * self.subsample**2

    @property
    def time(self,): return self.var.get('time')

    @property
    def pixel(self,): return self.var.get('pixel')

    @property
    def pixel_binned(self,): return self.pixel * self.subsample

    @property
    def detuning(self,): return self.var.get('detuning')

    @property
    def od_method(self,): return self.var.get('od_method')

    @property
    def sigmaf(self,): return self.var.get('sigmaf')

    @property
    def sigma(self,): return self.var.get('sigma', cst_Image_Class.sigma0 * self.sigmaf)

    @property
    def memory_saver(self,): return self.var.get('memory_saver')

    @property
    def lookup_table_version(self,): return self.var.get('lookup_table_version')
#####

##### Procedures
    '''Recalc Manager'''
    @property
    def recalc(self,): return self.var.get('recalc')

    '''Main Setter Function'''
    def set(self, **kwargs):
        if kwargs.get('refresh',False):
            self.var['recalc'] = [True] * len(self.recalc)
            return None
        keys = kwargs.keys()
        # recalc[0] is True if any of the keys in level 0 is provided and is different from current value
        recalc = [any([(j in keys) and (kwargs[j] != self.var.get(j,None)) for j in i]) for i in self.var['Level_Selector']]
        # Update self.var
        self.var = {**self.var, **kwargs}
        # If recalc[2] is True, then all that follows must also be true
        for i in range(len(recalc)):
            if recalc[i]:
                recalc[i+1:] = [True]*len(recalc[i+1:])
                break
        # self.recalc[0] is True if recalc[0] or self.recalc[0] was already True
        self.var['recalc'] = [recalc[i] or self.recalc[i] for i in range(len(recalc))]

    '''Load Image into Memory == Crop, Subsample, Rotate ==> Store cropi, Ii_raw, If_raw'''
    def prep_image(self,):
        [If, Ii] = self.alldata
        for task in self.prep_order:
            if task == 'crop':
                cropi = imagedata.get_cropi(Ii, **self.cropset)  # Need to improve speed here, takes 50 ms, (99% of time spent at [XX, YY] = np.meshgrid(x, y))
                Ii = Ii[cropi]
                If = If[cropi]
            elif (task == 'rotate') and (self.rotate != 0):
                Ii = scipy.misc.imrotate(Ii, angle=self.rotate, interp=self.rotate_method) # Takes 250 ms
                If = scipy.misc.imrotate(If, angle=self.rotate, interp=self.rotate_method) # takes 250 ms
            elif (task == 'subsample') and (self.subsample != 1):
                Ii = smooth.subsample2D(Ii, bins=[self.subsample, self.subsample]) # 1 ms
                If = smooth.subsample2D(If, bins=[self.subsample, self.subsample]) # 1 ms
        self.var['If_raw'], self.var['Ii_raw'] = If, Ii
        self.var['recalc'][0] = False
        self.var['cropi'] = cropi

    '''Find alpha for background subtraction'''
    def border_gradient(self,):
        # If width is set to 0
        if self.bg_width == 0:
            self.var['alpha_Ii'] = np.ones_like(self.Ii_raw)
            self.var['recalc'][1] = False
            return None

        # Get slicer for the border
        data = self.If_raw / self.Ii_raw
        mask = np.ones_like(data)
        w = self.bg_width
        s = data.shape
        mask[w:s[0]-w, w:s[1]-w] = 0
        using = np.logical_and((mask==1) , (np.isfinite(data)) )

        # Get Data for fitting
        xx, yy = np.meshgrid(np.arange(s[1]), np.arange(s[0]))
        xx_f, yy_f, zz_f = (xx[using], yy[using], data[using])
        def poly_2d(xy, b, m1=0, m2=0):
            return b + m1*xy[0] + m2*xy[1]

        # Fit
        guess = [1e-1]
        if self.bg_order == 1: guess = [1e-1, 1e-5, 1e-5]
        fitres, fiterr = scipy.optimize.curve_fit(poly_2d, (xx_f, yy_f), zz_f, p0=guess)
        self.var['alpha_Ii'] = poly_2d((xx, yy), *fitres)
        self.var['recalc'][1] = False

        # Warning for correction larger than 10%
        if abs(np.mean(self.var['alpha_Ii'])-1) >= 0.1:
            print('WARNING! Background correction is larger than 10%. Imagename {}'.format(self.name))

    '''Compute Optical Density'''
    def optical_density(self,):
        method = self.od_method
        if method in ['table','dBL']: self.var['od'] = interp_od(self.si, self.sf, self.time)
        elif method in ['sBL']: self.var['od'] = - np.log(self.sf / self.si) + self.si - self.sf
        else: self.var['od'] = - np.log(self.sf / self.si)
        self.var['recalc'][2] = False

#####

##### Plots
    def imshow(self, ax=None):
        if ax is None: _, ax = plt.subplots(figsize=(4,4))

        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="8%", pad=0.05)
        fig1 = ax.get_figure()
        fig1.add_axes(ax_cb)
        im = ax.imshow(self.app, origin='lower')
        plt.colorbar(im, cax=ax_cb)
        ax.set_axis_off()
        ax.set(title='Atoms/Pixel')


    def plot_crop(self, ax=None):
        alldata = self.alldata
        w = self.bg_width
        s = self.Ii_raw.shape
        cropi = self.cropi

        # Prepare Box
        x = [cropi[1].start,cropi[1].start,cropi[1].stop,cropi[1].stop,cropi[1].start]
        y = [cropi[0].start,cropi[0].stop,cropi[0].stop,cropi[0].start,cropi[0].start]
        x.extend([x[2],x[3],x[1]])
        y.extend([y[2],y[3],y[1]])

        try: ax[1]
        except: ax = plt.subplots(figsize=(10,4), ncols=2)[1]

        # Plots
        divider = make_axes_locatable(ax[0])
        ax_cb = divider.new_horizontal(size="8%", pad=0.05)
        fig1 = ax[0].get_figure()
        fig1.add_axes(ax_cb)
        im = ax[0].imshow(np.log(alldata[1] / alldata[0]), clim = [self.od_raw.min(), self.od_raw.max()], origin='lower')
        plt.colorbar(im, cax=ax_cb)
        ax[0].plot(x, y, 'w-', alpha=0.5)
        ax[0].set(title='Bare Image')

        divider = make_axes_locatable(ax[1])
        ax_cb = divider.new_horizontal(size="8%", pad=0.05)
        fig1 = ax[1].get_figure()
        fig1.add_axes(ax_cb)
        im = ax[1].imshow(self.od_raw, origin='lower')
        plt.colorbar(im, cax=ax_cb)
        ax[1].set(title='Cropped, Rotated, Subsampled')
        ax[1].plot([w, w, s[1] - w, s[1] - w, w], [w, s[0] - w, s[0] - w, w, w], 'w-')
        ax[1].set(xlim=[0,s[1]], ylim=[0,s[0]])
        fig1.tight_layout()

    def plot_border_gradient(self,):
        data = self.If_raw / self.Ii_raw
        s = data.shape
        w = self.bg_width
        alpha_Ii = self.alpha_Ii

        fig, ax = plt.subplots(figsize=(8, 5), nrows=2, ncols=3)
        ax[0,0].imshow(self.od_raw, aspect='auto', origin='lower')
        ax[0,0].plot([w, w, s[1] - w, s[1] - w, w], [w, s[0] - w, s[0] - w, w, w], 'w-')
        ax[0,0].set_axis_off()
        ax[0,0].set(title='BG Width Boundary')
        if w != 0:
            ax[0,2].plot(np.nanmean(alpha_Ii[0:w, :], axis=0),'k-')
            ax[0,2].plot(np.nanmean(data[0:w,:], axis=0), '.', markersize=2)
            ax[0,2].set(title='top')
            ax[1,0].plot(np.nanmean(alpha_Ii[:, 0:w], axis=1),'k-')
            ax[1,0].plot(np.nanmean(data[:,0:w], axis=1), '.', markersize=2)
            ax[1,0].set(title='left')
            ax[1,1].plot(np.nanmean(alpha_Ii[:, -w:], axis=1),'k-')
            ax[1,1].plot(np.nanmean(data[:,-w:], axis=1), '.', markersize=2)
            ax[1,1].set(title='right')
            ax[1,2].plot(np.nanmean(alpha_Ii[-w:, :], axis=0),'k-')
            ax[1,2].plot(np.nanmean(data[-w:,:], axis=0), '.', markersize=2)
            ax[1,2].set(title='bottom')

        divider = make_axes_locatable(ax[0,1])
        ax_cb = divider.new_horizontal(size="8%", pad=0.05)
        fig.add_axes(ax_cb)
        im = ax[0,1].imshow((self.alpha_Ii - 1)*100, aspect='auto', origin='lower')
        plt.colorbar(im, cax=ax_cb)
        ax[0,1].set_axis_off()
        ax[0,1].set(title='(alpha_Ii - 1) * 100')

        fig.tight_layout()
#####


####################################################################################
############################ XSectionHybrid ########################################

# XSectionHybrid Class Definition
# Compute 1d density from atoms per pixel

# for an ellipse with horizontal length a and vertical length b. Area from -A to A in horizontal
@np.vectorize
def area_partial_ellipse(A, a, b=None):
    if b is None: b = a
    if A >= a: return np.pi*a*b
    return 2*A*b*np.sqrt(1-(A/a)**2) + 2*a*b*np.arcsin(A/a)

'''Cross Section Hybrid'''
class XSectionHybrid:
    '''
    Take od and compute center and radius for each z_pixel

    Inputs :
        1) od
        2) bg_width
        3) ellipticity
        4) xsec_extension_method = 'polyN' or 'linear
        5) xsec_slice_width = 5, number of pixels to average for circle fits
        6) xsec_fit_range = 1, multiplies the fitted gaussian sigma

    Procedure:
        1) Approximate cloud center and fit gaussian
        2) Fit circles in provided region; store left, right, center
        3) Extend the results to entire image
        4) Functions for left, right, center, radius, area, sub_area(l, r)
    '''

    def __init__(self, od, ellipticity=1, extension='default', slice_width=4, fit_range=1.75):
        # Process Inputs
        self.data = od
        self.var = dict(ellipticity=ellipticity, extension=extension,
                        slice_width=slice_width, fit_range=fit_range)
        # Get fitting range
        self.z_edges, self.z_center = self.circle_fitting_range()
        # Fit circles
        self.center_fit, self.radius_fit = self.fit_circles()
        # Extrapolate
        self.z, self.center, self.radius = self.extrapolate()

    def circle_fitting_range(self, ):
        '''
        Get approximate center and radius by a Thomas-Fermi fit
        Use region radius*fit_range for circle fitting
        '''
        # Inputs
        slice_width = self.var['slice_width']
        fit_range = self.var['fit_range']
        # Integrate hybrid
        c = Curve(y = np.nanmean(self.data, axis=1))
        c.removenan()
        # Fit Gaussian
        def fitfun(x, x0, sigma, amp, a0):
            return np.exp(- (x-x0)**2 / (2*sigma**2)) * amp + a0
        guess = [c.x.shape[0]/2, 34, np.max(c.y), np.mean(c.y[0:10])]
        fitres = c.fit(fitfun, guess, plot=False)[0]
        center, radius_use = round(fitres[0]), round(fitres[1] * fit_range)
        # z_edges and z_center arrays
        z_edges = np.arange(center - radius_use, center + radius_use, slice_width, dtype=np.int)
        z_center = z_edges[0:-1] + (z_edges[1]-z_edges[0])/2.0 - 0.5 # half because slice doesn't include end point
        return (z_edges, z_center)

    def fit_circles(self):
        '''
        Fit circles to the range specified
        Measure center and radius at each point
        '''
        # Inputs
        z_center, z_edges = self.z_center, self.z_edges
        # Prepare arrays
        center, radius = np.zeros_like(z_center), np.zeros_like(z_center)
        # Replace infinities with nan
        use_data = self.data.copy()
        use_data[~np.isfinite(use_data)] = np.nan
        # Fit gaussian to the central slice to get initial guesses for circle fits
        def fitfun(x, x0, sigma, amp, a0):
            return np.exp(- (x-x0)**2 / (2*sigma**2)) * amp + a0
        i = len(z_center) // 2
        c = Curve(y = np.nanmean(use_data[z_edges[i]:z_edges[i+1],:], axis=0))
        c.removenan()
        guess = (c.x.shape[0] / 2, c.x.shape[0] / 5, np.max(c.y), np.mean(c.y[0:10]))
        fitres_gauss = c.fit(fitfun, guess, plot=False)[0]
        # Fit circles to each slices
        for i in range(self.z_center.size):
            c = Curve(y = np.nanmean(use_data[z_edges[i]:z_edges[i+1],:], axis=0))
            c.removenan()
            guess = (fitres_gauss[0], fitres_gauss[1]*1.75, np.max(c.y), fitres_gauss[3])
            fitres = c.fit(self.fitfun_circle, guess, plot=False)[0]
            if fitres[0] == guess[0]:
                center[i], radius[i] = np.nan, np.nan
            else: center[i], radius[i] = fitres[0], fitres[1]
        # return results
        return (center, radius)

    def extrapolate(self):
        '''
        Extrapolate the fitted center and radius
        using either polyN or splineN method
        '''
        # Inputs
        method = self.var['extension']
        z_center_fit, center_fit, radius_fit = self.z_center, self.center_fit, self.radius_fit
        # Empty arrays for storage
        z, center, radius = np.arange(self.data.shape[0]), np.arange(self.data.shape[0]), np.arange(self.data.shape[0])
        c_center = Curve(z_center_fit, center_fit)
        c_center.removenan()
        c_radius = Curve(z_center_fit, radius_fit)
        c_radius.removenan()
        # Linearly extend the center
        fitres = np.poly1d(np.polyfit(*c_center.plotdata, deg=1))
        center = fitres(z)
        # polyN
        if method[0:4] == 'poly':
            fitres = np.poly1d(np.polyfit(*c_radius.plotdata, deg=int(method[4:])))
            radius = fitres(z)
            radius[z<z_center_fit[0]] = fitres(z_center_fit[0])
            radius[z>z_center_fit[-1]] = fitres(z_center_fit[-1])
        elif method == 'linear':
            fitres = np.poly1d(np.polyfit(*c_radius.plotdata, deg=1))
            radius = fitres(z)
        else:
            def fitfun(x, a0, a1=0, a2=0): return a0 + a1*x + a2*x**2
            fitres = c_radius.fit(fitfun, [np.mean(c_radius.y), 0, 0], noise=1, plot=False)[0]
            radius = fitfun(z, *fitres)
            radius[z<z_center_fit[0]] = fitfun(z_center_fit[0], *fitres)
            radius[z>z_center_fit[-1]] = fitfun(z_center_fit[-1], *fitres)
        # Return
        return (z, center, radius)

    '''
    Useful calls to get center, radius, left, right, area, and sub_area
    '''
    def get_center(self, z):
        z = np.array(z, dtype=np.int32)
        return self.center[z]

    def get_radius(self, z):
        z = np.array(z, dtype=np.int32)
        return self.radius[z]

    def get_left(self, z):
        return self.get_center(z) - self.get_radius(z)

    def get_right(self, z):
        return self.get_center(z) + self.get_radius(z)

    def get_area(self, z):
        return np.pi * self.get_radius(z)**2 * self.var['ellipticity']

    def get_subarea(self, z, l, r):
        a = self.get_radius(z)
        b = a * self.var['ellipticity']
        Al = self.get_center(z) - l
        Ar = r - self.get_center(z)

        # Check for errors
        if np.any(Al <= 0) or np.any(Ar <= 0):
            print("Illegal left and right points given to XSectionHybrid.get_subarea. Returned total area.")
            return self.get_area(z)

        return area_partial_ellipse(Al,a,b)/2 + area_partial_ellipse(Ar,a,b)/2


    def infoplot(self, axs=None, left=None, right=None):
        '''
        Useful information plots: data with fitted center and radius + extrapolation
        Ability to plot on provided axes
        '''
        if axs is None:
            fig, axs = plt.subplots(figsize=(5,5), nrows=2)
        axs[0].imshow(self.data.T, cmap='viridis', aspect='auto', origin='lower')
        axs[0].plot(self.z, self.center,'w--',alpha=0.5)
        axs[0].plot(self.z, self.center - self.radius,'w--',alpha=0.5)
        axs[0].plot(self.z, self.center + self.radius,'w--',alpha=0.5)
        axs[0].scatter(self.z_center, self.center_fit - self.radius_fit,color='white', s=2)
        axs[0].scatter(self.z_center, self.center_fit + self.radius_fit,color='white', s=2)
        axs[0].scatter(self.z_center, self.center_fit,color='white', s=2)
        axs[0].set(xlim=(self.z[0],self.z[-1]))
        axs[0].set_axis_off()

        if left is not None and right is not None:
            axs[0].plot(left,'r-',alpha=0.7)
            axs[0].plot(right,'r-',alpha=0.7)

        axs[1].scatter(self.z_center, self.radius_fit,color='red')
        axs[1].plot(self.z, self.radius,'k')
        axs[1].set(xlim=(self.z[0],self.z[-1]), ylabel='Radius')

    def fitfun_circle(self, x, x0, rad, amp, a0):
        y = 1 - ((x - x0) / rad) ** 2
        y[y <= 0] = 0
        y[y > 0] = np.sqrt(y[y > 0]) * amp
        y += a0
        return y



####################################################################################
################################ Density_Generatos #################################


# Standalone Density Generator

# Load pre-compiled data
p_ = roots1.getpath('Projects','Data','EoS','Mark_Density_EoS_Extended_Data4Python.p')
if not os.path.isfile(p_):
    print("Downloading Database -- Might take some time!")
    url = "https://www.dropbox.com/s/abxs9yarrgohzy8/Mark_Density_EoS_Extended_Data4Python.p?dl=1"
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    # Create folder
    os.makedirs(os.path.split(p_)[0], exist_ok=True)
    with open(p_, "wb") as f :
        f.write(data)
precompiled_data_EoS_Density_Generator = pickle.load( open( p_, "rb" ) )

# Other needed functions
constants_for_density_ = constants.cst()
constants_for_density_.c1 = ((constants_for_density_.twopi * constants_for_density_.hbar**2)/(constants_for_density_.mass))**(1/2)
constants_for_density_.c2 = 1/(6*constants_for_density_.pi**2) * (2*constants_for_density_.mass/constants_for_density_.hbar**2)**(3/2)
constants_for_density_.xi = 0.37
constants_for_density_.virial_coef = [1, 3*2**(1/2)/8, -0.29095295, 0.065]

def thermal_wavelength(kT):
    return constants_for_density_.c1 / (kT)**(1/2)

@np.vectorize
def density_ideal(kT, mu):
    if kT == 0:
        if mu <= 0:
            print('Density is undefined for negative mu and zero temperature')
            return 0
        return cst_.c2 * (mu)**(3/2)
    return thermal_wavelength(kT)**(-3) * functions.FermiFunction(m=3/2, logz=mu/kT)

@np.vectorize
def density_virial(kT, mu):
    if kT == 0: return 0
    return kT / thermal_wavelength(kT)**3 * (constants_for_density_.virial_coef[0]*1/kT*np.exp(1*mu/kT) + constants_for_density_.virial_coef[1]*2/kT*np.exp(2*mu/kT) + constants_for_density_.virial_coef[2]*3/kT*np.exp(3*mu/kT) + constants_for_density_.virial_coef[3]*4/kT*np.exp(4*mu/kT))

# Setup function
@np.vectorize
def density_unitary(kT, mu):
    # Zero T
    if kT < 0:
        return 0
    if kT == 0:
        return constants_for_density_.EF2n(mu/constants_for_density_.xi, neg=True)
    if mu/kT > 4:
        return constants_for_density_.EF2n(mu / precompiled_data_EoS_Density_Generator[0](kT/mu), neg=True)
    if mu/kT > -0.5:
        return constants_for_density_.EF2n(mu / precompiled_data_EoS_Density_Generator[1](mu/kT), neg=True)
    return density_virial(kT, mu)




####################################################################################
################################ Hybrid_Image ######################################


cst_Hybrid_Image = constants.cst()

Default_Hybrid_Image = dict(ellipticity=1, xsec_extension='default',xsec_slice_width=4,
                            xsec_fit_range=1.75, xsec_override=False, trap_f=23.9,
                            radial_selection=0.7, trap_center_override=False, kind='unitary',
                            Tfit_lim=0.2, Tfit_guess_kT=0.5, Tfit_guess_mu0=1)

Level_Selector_Hybrid_Image = [['ellipticity','xsec_extension', 'xsec_slice_width','xsec_fit_range', 'xsec_override'],
                               ['fudge','sigmaf','sigma','pixel','trap_f','radial_selection','trap_center_override'],
                               ['kind','Tfit_lim', 'Tfit_guess_kT', 'Tfit_guess_mu0']]

class Hybrid_Image(Image):
    def __init__(self, name=None, path=None, od=None, **kwargs):
        # Initialize the complete Image Object
        super(Hybrid_Image, self).__init__(name=name, path=path, od=od, **kwargs)

        # Addons
        self.var = {**self.var, **Default_Hybrid_Image, **kwargs}
        self.LevelAdder = len(self.var['Level_Selector'])
        self.var['Level_Selector'] = self.var['Level_Selector'] + list(Level_Selector_Hybrid_Image)
        self.var['recalc'] = self.var['recalc'] + [True]*len(Level_Selector_Hybrid_Image)

    # New Properties
    @property
    def ellipticity(self,): return self.var.get('ellipticity')

    @property
    def xsec_extension(self,): return self.var.get('xsec_extension')

    @property
    def xsec_slice_width(self,): return self.var.get('xsec_slice_width')

    @property
    def xsec_fit_range(self,): return self.var.get('xsec_fit_range')

    @property
    def xsec(self,):
        if (self.var['xsec_override'] is not False):
            xsec_override = self.var['xsec_override']
            xsec_override.data = self.od
            return xsec_override
        if self.recalc[0 + self.LevelAdder] or ('xsec' not in self.var.keys()):
            self.compute_xsec()
        return self.var.get('xsec')

    @property
    def trap_f(self,): return self.var.get('trap_f')

    @property
    def trap_w(self,): return 2 * np.pi * self.trap_f

    @property
    def radial_selection(self,): return self.var.get('radial_selection')

    @property
    def trap_center(self,):
        if self.var['trap_center_override'] is not False:
            return self.var['trap_center_override']
        if ('trap_center' not in self.var.keys()) or self.recalc[1 + self.LevelAdder]:
            self.compute_nz()
        return self.var['trap_center']

    @property
    def z(self,): return (np.arange(self.app.shape[0]) - self.trap_center) * self.pixel_binned

    @property
    def u(self,): return 0.5*cst_Hybrid_Image.mass*self.trap_w**2*self.z**2

    @property
    def n(self,):
        if ('n' not in self.var.keys()) or self.recalc[1 + self.LevelAdder]:
            self.compute_nz()
        return self.var['n']

    @property
    def N(self,):
        if ('N' not in self.var.keys()) or self.recalc[1 + self.LevelAdder]:
            self.compute_nz()
        return self.var['N']

    @property
    def nz(self,): return Curve(x=self.z, y=self.n, xscale=1e-6, yscale=1e18)

    @property
    def nu(self,): return Curve(x=self.u, y=self.n, xscale=1e3*cst_Hybrid_Image.h, yscale=1e18)

    @property
    def EFu(self,): return Curve(x=self.u, y=cst_Hybrid_Image.n2EF(self.n, neg=True), xscale=1e3*cst_Hybrid_Image.h, yscale=1e3*cst_Hybrid_Image.h)

    @property
    def ku(self,):
        EFu = self.EFu.sortbyx().subsample(bins=2)
        ku = EFu.diff(method='poly', order=1, points=4)
        return Curve(x=ku.x, y=-ku.y, xscale=ku.xscale).subsample(bins=2)

    @property
    def kz_u(self,):
        ku = self.ku
        z = (2 * ku.x / cst_Hybrid_Image.mass / self.trap_w**2 )**(1/2)
        return Curve(x=np.concatenate([np.flipud(-z), z]), y=np.concatenate([np.flipud(ku.y), ku.y]), xscale=1e-6, yscale=1)

    @property
    def kind(self,): return self.var.get('kind')

    @property
    def Tfit_lim(self,): return self.var.get('Tfit_lim')

    @property
    def Tfit_guess(self,): return [self.var.get('Tfit_guess_kT'), self.var.get('Tfit_guess_mu0')]

    @property
    def T_kHz(self,):
        if ('T_kHz' not in self.var.keys()) or self.recalc[2 + self.LevelAdder]:
            self.compute_temperature()
        return self.var['T_kHz']

    @property
    def mu0_kHz(self,):
        if ('mu0_kHz' not in self.var.keys()) or self.recalc[2 + self.LevelAdder]:
            self.compute_temperature()
        return self.var['mu0_kHz']

    @property
    def TTF(self,):
        return self.T_kHz * 1e3 * cst_Hybrid_Image.h / cst_Hybrid_Image.n2EF(self.n, neg=True)

    @property
    def TTF_center(self,):
        fitfun, fitres = self.var['Tfit_info']
        n_center = np.max(fitfun(self.nz.x, *fitres))
        return self.T_kHz * 1e3 / cst_Hybrid_Image.n2EFHz(n_center)

    @property
    def Tfit_residual(self,):
        return np.nanmean(np.abs(self.var['Tfit_info'][0](self.nz.x, *self.var['Tfit_info'][1]) - self.n))

    # Procedures
    def compute_xsec(self,):
        xsec = XSectionHybrid(self.od, ellipticity=self.ellipticity, extension=self.xsec_extension,
                                   slice_width=self.xsec_slice_width, fit_range=self.xsec_fit_range)
        self.var['recalc'][0 + self.LevelAdder] = False
        self.var['xsec'] = xsec

    def compute_nz(self,):
        # Compute n(i)
        i = np.arange(self.app.shape[0])
        l = i*0
        r = i*0 + self.app.shape[1] - 1

        if (self.radial_selection == 1) or (self.radial_selection==0):
            N = np.nansum(self.app, axis=1)
            n = N / (self.xsec.get_area(i) * self.pixel_binned**3)
        elif self.radial_selection < 1:
            l = np.array(np.round(self.xsec.get_center(i) - self.xsec.get_radius(i) * self.radial_selection), dtype=np.int)
            r = np.array(np.round(self.xsec.get_center(i) + self.xsec.get_radius(i) * self.radial_selection), dtype=np.int)
            N = np.array([np.nansum(self.app[j, l[j]:1+r[j]]) for j in i])
            n = N / (self.xsec.get_subarea(i, l-0.5, r+0.5) * self.pixel_binned**3)
        elif self.radial_selection > 1:
            l = np.array(np.round(self.xsec.get_center(i) - self.radial_selection), dtype=np.int)
            r = np.array(np.round(self.xsec.get_center(i) + self.radial_selection), dtype=np.int)
            N = np.array([np.nansum(self.app[j, l[j]:1+r[j]]) for j in i])
            n = N / (self.xsec.get_subarea(i, l-0.5, r+0.5) * self.pixel_binned**3)
        # Note that l-0.5 and r+0.5 are used because of the way integrals over pixels work out. Has been tested!
        ni = Curve(x=i, y=n)

        # Find Center i0
        def fitfun(x, x0, rad, amp, a0):
            y = np.real((1-((x-x0)/(rad))**2)**(3/2))
            y[~np.isfinite(y)] = 0
            return amp*y + a0
        guess = [ni.x[ni.y==ni.maxy][0], ni.x.size/5, ni.maxy, np.mean(ni.y[0:5])]
        fitres = ni.fit(fitfun, guess, plot=False)[0]

        # Store
        self.var['trap_center'] = fitres[0]
        self.var['n'] = n
        self.var['N'] = N
        self.var['radial_selection_info'] = (l, r)
        self.var['recalc'][1 + self.LevelAdder] = False

    def compute_temperature(self,):
        nz = self.nz
        if (self.kind == 'unitary') or (self.kind == 'balanced'):
            def fitfun(z, kT, mu0, a0=0, z0=0):
                kT *= 1e3*cst_Hybrid_Image.h
                mu0 *= 1e3*cst_Hybrid_Image.h
                mu = mu0 - 1/2 * cst_Hybrid_Image.mass * self.trap_w**2 * (z-z0*1e-6)**2
                return density_unitary(kT, mu) + a0
        elif (self.kind == 'ideal') or (self.kind == 'polarized'):
            def fitfun(z, kT, mu0, a0=0, z0=0):
                kT *= 1e3*cst_Hybrid_Image.h
                mu0 *= 1e3*cst_Hybrid_Image.h
                mu = mu0 - 1/2 * cst_Hybrid_Image.mass * self.trap_w**2 * (z-z0*1e-6)**2
                return density_ideal(kT, mu) + a0
        fitres = nz.fit(fitfun, [self.Tfit_guess[0], self.Tfit_guess[1], np.mean(nz.y[0:5])], plot=False, ylim=(-np.inf, self.Tfit_lim*1e18))[0]

        # Store
        self.var['T_kHz'] = fitres[0]
        self.var['mu0_kHz'] = fitres[1]
        self.var['Tfit_info'] = (fitfun, fitres)
        self.var['recalc'][2 + self.LevelAdder] = False


    # Plots
    def plot_hybrid_info(self,ulim=10, zlim=250, klim=(-0.5,3.5), output=False):
        fig = plt.figure(figsize = (12,7))
        ax1 = fig.add_subplot(4,3,1)
        ax2 = fig.add_subplot(4,3,4)
        ax3 = fig.add_subplot(2,3,2)
        ax4 = fig.add_subplot(2,3,3)
        ax5 = fig.add_subplot(4,3,7)
        ax6 = fig.add_subplot(4,3,10)
        ax7 = fig.add_subplot(2,3,5, sharex=ax3)
        ax8 = fig.add_subplot(2,3,6, sharex=ax4, sharey=ax7)
        # plots
        ax3.plot(*self.nz.plotdata)
        ax3.plot(self.nz.plotdata[0], self.nz.x*0, 'k--', alpha=0.5)
        ax3.plot([0,0], [0, self.nz.maxy/self.nz.yscale], 'k--', alpha=0.5)
        ax4.plot(*self.EFu.plotdata)
        ax4.plot(self.EFu.plotdata[0], self.EFu.x*0, 'k--', alpha=0.5)
        ax3.set(xlabel=r'z [$\mu m$]', ylabel=r'n [$\mu m ^{-3}$]', title='Density', xlim=[-zlim, zlim])
        ax4.set(xlabel=r'u [kHz]', ylabel=r'$E_F$ [kHz]', title='Fermi Energy', xlim=[0, ulim])
        self.xsec.infoplot([ax1, ax2], *self.var['radial_selection_info'])
        ax7.plot(*self.kz_u.plotdata)
        ax7.plot(self.kz_u.plotdata[0], self.kz_u.x*0+1, 'k--', self.kz_u.plotdata[0], self.kz_u.x*0+1/0.37, 'k--', self.kz_u.plotdata[0], self.kz_u.x*0, 'k--', alpha=0.5)
        ax7.set(xlabel=r'z [$\mu$ m]', ylabel=r'$\kappa / \kappa_0$', ylim=klim)
        ax8.plot(*self.ku.plotdata)
        ax8.plot(self.ku.plotdata[0], self.ku.x*0+1, 'k--', self.ku.plotdata[0], self.ku.x*0+1/0.37, 'k--', self.ku.plotdata[0], self.ku.x*0, 'k--', alpha=0.5)
        ax8.set(xlabel=r'u [kHz]', title=self.name)
        axs = (ax1, ax2, ax5, ax6, ax3, ax4, ax7, ax8)
        fig.tight_layout(pad=0.1, h_pad=0, w_pad=0)
        if output: return (fig, axs)

    def plot_hybrid_temp_info(self, ulim=10, zlim=250, klim=(-0.5,3.5), Tlim=None, Tstep = None):
        fig, ax = self.plot_hybrid_info(ulim=ulim, zlim=zlim, klim=klim, output=True)

        nz = self.nz
        TTF = self.TTF
        fitfun, fitres = self.var['Tfit_info']
        nz_fit = Curve(x = nz.x, y = fitfun(nz.x, *fitres), xscale=nz.xscale, yscale=nz.yscale)
        # Plot fitted profile
        ax[4].plot(*nz_fit.plotdata, alpha=0.75)
        if nz.maxy >= self.Tfit_lim*1e18: ax[4].plot(nz.plotdata[0], nz.x*0 + self.Tfit_lim, 'k--', alpha=0.5)
        ax[4].set(title='Density, Center T/TF {:.2f}'.format(self.TTF_center))

        # Plot Residuals
        ax[2].plot(nz.plotdata[0], 100 * (nz.y - nz_fit.y) / nz.yscale)
        ax[2].plot(nz.plotdata[0], nz.x*0, 'k--', alpha=0.5)
        ax[2].set(xlim=[-zlim,zlim], ylabel=r'Res [100 $\times$ $\mu m ^{-3}$]', xlabel=r'z [$\mu$m]')
        ax[2].set(title='Offset {:.2f} [100 x um^-3]'.format(fitres[2]/1e18*100))

        # Information
        ax[6].set(title=r'T = {:.2f} kHz, $\mu_0$ = {:.2f} kHz'.format(fitres[0], fitres[1]))

        # Histogram of N(T/TF)
        if fitres[0] <= 0.02: return None
        if Tlim is None: Tlim = TTF[np.abs(nz.x) == np.abs(nz.x).min()][0] * 3
        if Tstep is None: Tstep = np.round(TTF[np.abs(nz.x) == np.abs(nz.x).min()][0]/10, 3)
        c = Curve(TTF, self.N)
        c.removenan()
        c = c.sortbyx().trim(xlim=[0, Tlim]).binbyx(step=Tstep, sects=[0,Tlim], func=np.nansum, center_x=True)
        ax[3].bar(left = c.x - Tstep/2, height = c.y / np.nansum(c.y * Tstep), width=Tstep)
        ax[3].plot([0.17]*2, [0, 1], 'k--', alpha=0.5)
        ax[3].set(xlabel=r'$T/T_F$', ylabel=r'Fraction of Atoms', xlim=(0, Tlim), ylim=(0, np.nanmax(c.y / np.nansum(c.y * Tstep))*1.1))







####################################################################################
################################ Curve #############################################

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

    def __init__(self, x=None, y=np.array([]), **kwargs):
        if x is None: x = np.arange(y.size)
        self.var = kwargs
        self.var['x'] = x.copy()
        self.var['y'] = y.copy()

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
    def fitusing(self):
        return self.var.get('fitusing', None)

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
    def plotdata(self):
        return (self.x / self.xscale, self.y / self.yscale)

    @property
    def xscale(self):
        return self.var.get('xscale',1)

    @property
    def yscale(self):
        return self.var.get('yscale', 1)

    @property
    def miny(self): return np.min(self.y)

    @property
    def maxy(self): return np.max(self.y)

    @property
    def minx(self): return np.min(self.x)

    @property
    def maxx(self): return np.max(self.x)

    ### High level methods ###
    def __call__(self, xi):
        return np.interp(xi, self.x[self.sorti], self.y[self.sorti])

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

    def chop(self,xlim=None,ylim=None):
        return self.trim(xlim, ylim)

    def subset(self, xlim=None, ylim=None):
        return self.trim(xlim, ylim)

    def trim(self,xlim=None,ylim=None):
        # Prepare using
        using = np.array(np.ones_like(self.x), np.bool)
        if xlim is not None:
            using[self.x < xlim[0]] = False
            using[self.x > xlim[1]] = False
        if ylim is not None:
            using[self.y < ylim[0]] = False
            using[self.y > ylim[1]] = False
        if np.sum(using) <= 2:
            using = np.array(np.ones_like(self.x), np.bool)
            print("X and Y limits given leads to too little points. All are being used")
        return self.copy(self.x[using], self.y[using])

    def sortbyx(self):
        return self.copy(self.x[self.sorti], self.y[self.sorti])

    def binbyx(self, **kwargs):
        return self.copy(*smooth.binbyx(self.x, self.y, **kwargs))

    def subsample(self, bins=2):
        return self.copy(*smooth.subsampleavg(self.x, self.y, bins=bins))

    def diff(self, **kwargs):
        method = kwargs.get('method', 'poly')
        if method == 'poly':
            dydx = calculus.numder_poly(self.x, self.y, order=kwargs.get('order', 1), points=kwargs.get('points', 1))
        elif method == 'central2':
            dydx = np.gradient(self.y, self.dx, edge_order=2)
        return self.copy(self.x, dydx)

    def removenan(self):
        self.var['x'] = self.x[np.isfinite(self.y)]
        self.var['y'] = self.y[np.isfinite(self.y)]

    def copy(self, x=None, y=None):
        if x is None: x = self.x
        if y is None: y = self.y
        return Curve(x=x, y=y, xscale=self.xscale, yscale=self.yscale)

    def fit(self, fitfun, guess, plot=False, pts=1000, noise=None, loss='cauchy', bounds=(-np.inf, np.inf), xlim=None, ylim=None):
        # Prepare using
        using = np.array(np.ones_like(self.x), np.bool)
        if xlim is not None:
            using[self.x<xlim[0]] = False
            using[self.x>xlim[1]] = False
        if ylim is not None:
            using[self.y<ylim[0]] = False
            using[self.y>ylim[1]] = False
        if np.sum(using) <= len(guess):
            using = np.array(np.ones_like(self.x), np.bool)
            print("X and Y limits given leads to too little points. All are being used")

        # Fit
        if noise is None:
            from scipy.optimize import curve_fit
            try:
                fitres, fiterr = curve_fit(fitfun, self.x[using], self.y[using], p0=guess, bounds=bounds)
                fiterr = np.sqrt(np.diag(fiterr))
            except RuntimeError as err:
                fitres = guess
                fiterr = guess
                print("CAN'T FIT, Returning Original Guess: Details of Error {}".format(err))
        else:
            from scipy.optimize import least_squares
            try:
                fitfun_ = lambda p: fitfun(self.x[using], *p) - self.y[using]
                fitres_ = least_squares(fun=fitfun_, x0=guess, loss=loss, f_scale=noise, bounds=bounds)
                fitres = fitres_.x
                fiterr = np.zeros_like(guess) * np.nan
            except RuntimeError as err:
                fitres = guess
                fiterr = np.zeros_like(guess) * np.nan
                print("CAN'T FIT, Returning Original Guess: Details of Error {}".format(err))

        yfit = fitfun(self.x, *fitres)
        xfitplot = np.linspace(np.min(self.x), np.max(self.x), pts)
        yfitplot = fitfun(xfitplot, *fitres)
        # Save results in var
        self.var['fitusing'] = using
        self.var['yfit'] = yfit
        self.var['xyfitplot'] = (xfitplot, yfitplot)
        self.var['fitres'] = fitres
        self.var['fiterr'] = fiterr
        # Plot and display
        if plot:
            # Plot
            plt.figure(figsize=(5, 5))
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0))
            ax1.plot(*self.xyfitplot,'k-')
            ax1.plot(self.x, self.y, 'g.')
            ax1.plot(self.x[using], self.y[using],'r.')
            ax2.plot(self.x, self.y-self.yfit,'g.')
            ax2.plot(self.x[using], self.y[using] - self.yfit[using], 'r.')
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


###################################################################################

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
            self.var['od'] = (self.rawod * self.cst.ODf) + ((self.woa - self.wa) / self.cst.Nsat)
        return self.var['od']

    @property
    def rawod(self):
        if 'rawod' not in self.var.keys():
            self.var['rawod'] = imagedata.get_od(self.wa, self.woa, rawod=True)
        return self.var['rawod']

    @property
    def fixod(self):
        if 'fixod' not in self.var.keys():
            rawod = imagedata.get_od(self.wa, self.woa, width=self.var.get('trouble_pts_width', 5), rawod=False)
            self.var['fixod'] = (rawod * self.cst.ODf) + ((self.woa - self.wa) / self.cst.Nsat)
        return self.var['fixod']

    @property
    def ncol(self):
        return self.od / self.cst.sigma

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
        cropi = imagedata.get_cropi(self.od, **kwargs)
        if kwargs.get('plot',False):
            x = [cropi[1].start,cropi[1].start,cropi[1].stop,cropi[1].stop,cropi[1].start]
            y = [cropi[0].start,cropi[0].stop,cropi[0].stop,cropi[0].start,cropi[0].start]
            fig, ax = plt.subplots(figsize=(7,3),ncols=2)
            ax[0].imshow(self.od, cmap='viridis', clim=kwargs.get('odlim',(0,2)), origin='lower')
            ax[0].plot(x, y,'w-')
            ax[0].set(xlim=[0,self.od.shape[1]], ylim=[0,self.od.shape[0]])
            ax[1].imshow(self.od[cropi], cmap='viridis', clim=kwargs.get('odlim',(0,2)), origin='lower')
            ax[1].set(xlim=[0, self.od[cropi].shape[1]], ylim=[0, self.od[cropi].shape[0]])
        return cropi

    # fix intensities
    def fixVaryingIntensities_AllOutside(self, xmin, xmax, ymin, ymax):
        # Define a crop region and find factor*woa
        (x,y) = self.xy
        cropi = np.logical_and.reduce((x>=xmin, x<=xmax, y>=ymin, y<=ymax))
        factor = np.sum(self.alldata[0][cropi==0]) / np.sum(self.alldata[1][cropi==0])
        self.var['factor_woa'] = factor
        # Recalculate wa, woa, od, fixod
        self.var['wa'] = self.alldata[0]
        self.var['woa'] = self.alldata[1] * self.var['factor_woa']
        if 'od' in self.var.keys(): del self.var['od']
        if 'fixod' in self.var.keys(): del self.var['fixod']
        if 'rawod' in self.var.keys(): del self.var['rawod']

    def fixVaryingIntensities_Box(self, cropi=None, **kwargs):
        # Define a crop region and find factor*woa
        (x,y) = self.xy
        if cropi is None: cropi = self.cropi(**kwargs)
        factor = np.sum(self.alldata[0][cropi]) / np.sum(self.alldata[1][cropi])
        self.var['factor_woa'] = factor
        # Recalculate wa, woa, od, fixod
        self.var['wa'] = self.alldata[0]
        self.var['woa'] = self.alldata[1] * self.var['factor_woa']
        if 'od' in self.var.keys(): del self.var['od']
        if 'fixod' in self.var.keys(): del self.var['fixod']
        if 'rawod' in self.var.keys(): del self.var['rawod']

    # Auto crop hybrid
    def autocrop_hybrid(self, plot=False, odlim=(0,2), border = 50):
        # along y
        c = Curve(y=np.nansum(self.od, axis=1))
        max_y = np.max(c.y[c.y.shape[0]//4:3*c.y.shape[0]//4])
        ind = np.argwhere(c.y == max_y)[0][0]
        guess = [c.x[ind], c.x.shape[0]/10, c.y[ind], c.y[ind]/10, c.y[ind]/100]
        fy = c.fit(functions.fitfun_TF_harmonic, guess, plot=False)[0]
        # along x
        c = Curve(y=np.nansum(self.od, axis=0))
        max_y = np.max(c.y[c.y.shape[0] // 4:3 * c.y.shape[0] // 4])
        ind = np.argwhere(c.y == max_y)[0][0]
        guess = [c.x[ind], c.x.shape[0] / 10, c.y[ind], c.y[ind] / 10, c.y[ind] / 100]
        fx = c.fit(functions.fitfun_TF_harmonic, guess, plot=False)[0]
        # Generate cropi
        center = (int(fx[0]),int(fy[0]))
        width = 2 * int(min(fx[1] * 2, center[0] - border, self.od.shape[1] - center[0] - border))
        height = 2 * int(min(fy[1] * 2, center[1] - border, self.od.shape[0] - center[1] - border))
        return self.cropi(center=center, width=width, height=height, plot=plot, odlim=odlim)

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


#################################################################################


class XSectionTop:
    '''
    Compute Cross sectional area for a hybrid image using circular fits

    Inputs:
        od     --  cropped od image (od is recommanded because i know that amplitude is in range 0 to ~3)
        yuse   --  the range of y indices to use for fitting circles. Use np.arange(start, stop, step).
                   use None (or don't provide) to auto generate it
        method --  method for extending fitted radii: linead (default), poly4, spline
        plot   --  True or False, a sample plot with analysis, default False
        odlim  --  clim for imshow for the od, default (0,2)
        yset   --  settings for auto yuse: (y step, fraction of R_TF to use), default (10,0.75)
        guess  --  guess for circle fit: (x_center, radius, amplitude, m, b), default (xlen/2, xlen/5, max)

    Useful properties and calls:
        self(y_indices) returns area for provided y_indices. (must be within od size range)
        self.rad
        self.area
        self.yall
    '''
    def __init__(self, od, yuse = None, method='linear', plot=False, odlim=(0,2), yset = (10, 0.75), guess = None):
        self.prepare(od, yuse, odlim, yset, guess)
        self.fitall()
        if method == 'spline': self.extend_spline()
        elif method == 'poly4': self.extend_poly4()
        else: self.extend_linear()

        if plot: self.infoplot()

    def __call__(self, y):
        # Make it an integer
        y = np.int32(np.round(y))
        return self.area[y]

    def prepare(self, od, yuse, odlim, yset, guess):
        # General things
        self.yuse = yuse
        self.od = od
        self.odlim = odlim
        self.guess = guess
        self.yset_ = yset
        if yuse is None: yuse = self.get_yuse()
        self.dy = yuse[1] - yuse[0]
        # ycen_ vs. xc_, r_, xl_, xr_, c_
        self.ycen_ = yuse[0:-1] + self.dy / 2
        self.xc_, self.r_ = np.zeros_like(self.ycen_), np.zeros_like(self.ycen_)
        self.c_ = [None] * self.xc_.shape[0]
        self.fitres_ = [None] * self.xc_.shape[0]
        self.fiterr_ = [None] * self.xc_.shape[0]
        # yall vs rad, area
        self.yall = np.arange(od.shape[0])

    def fitall(self):
        if self.guess is None:
            c = Curve(y=np.nanmean(self.od[self.yuse[0]:self.yuse[0 + 1], :], axis=0))
            self.guess = [c.x.shape[0] / 2, c.x.shape[0] / 5, np.max(c.y), 0, 0]
        for i, yc in enumerate(self.ycen_):
            c = Curve(y=np.nanmean(self.od[self.yuse[i]:self.yuse[i + 1], :], axis=0))
            c.removenan()
            fitres, fiterr = c.fit(self.fitfun, self.guess, plot=False)
            self.xc_[i] = fitres[0]
            self.r_[i] = fitres[1]
            self.c_[i] = c
            self.fitres_[i] = fitres
            self.fiterr_[i] = fiterr

        self.xl_ = self.xc_ - self.r_
        self.xr_ = self.xc_ + self.r_

    def get_yuse(self):
        c = Curve(y = np.nanmean(self.od,axis=1))
        c.removenan()
        # fit TF profile to this
        fitres, fiterr = c.fit(self.fitfun_TF, [c.x.shape[0] / 2, c.x.shape[0] / 4, np.max(c.y), np.max(c.y)/100, np.max(c.y)/100], plot=False)
        fitres[1] = fitres[1] * self.yset_[1]
        self.yuse = np.arange(int(fitres[0]-fitres[1]), int(fitres[0]+fitres[1]), self.yset_[0])
        return self.yuse

    def extend_linear(self):
        fitres = np.polyfit(self.ycen_, self.r_, deg=1)
        self.radfit = np.poly1d(fitres)
        self.rad = self.radfit(self.yall)
        self.area = np.pi * self.rad ** 2

    def extend_poly4(self):
        fitres = np.polyfit(self.ycen_, self.r_, deg=4)
        self.radfit = np.poly1d(fitres)
        self.rad = self.radfit(self.yall)
        self.rad[self.yall < self.ycen_[0]] = self.radfit(self.ycen_[0])
        self.rad[self.yall > self.ycen_[-1]] = self.radfit(self.ycen_[-1])
        self.area = np.pi * self.rad ** 2

    def extend_spline(self):
        tck = spintp.splrep(self.ycen_, self.r_, s=100)
        self.rad = spintp.splev(self.yall, tck, der=0)
        self.rad[self.yall < self.ycen_[0]] = spintp.splev(self.ycen_[0], tck, der=0)
        self.rad[self.yall > self.ycen_[-1]] = spintp.splev(self.ycen_[-1], tck, der=0)
        self.area = np.pi * self.rad ** 2

    def fitfun(self, x, x0, rad, amp, m, b):
        y = 1 - ((x - x0) / rad) ** 2
        y[y <= 0] = 0
        y[y > 0] = np.sqrt(y[y > 0]) * amp
        y += m * x + b
        return y

    def infoplot(self):
        # Figure
        fig = plt.figure(figsize=(8, 5))
        # Setup axes
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((3, 3), (1, 0), rowspan=2, colspan=2)
        axc = [None] * 3
        axc[0] = plt.subplot2grid((3, 3), (0, 2))
        axc[1] = plt.subplot2grid((3, 3), (1, 2))
        axc[2] = plt.subplot2grid((3, 3), (2, 2))
        # Plot od and measured edges
        ax1.scatter(self.ycen_, self.xl_, s=1, color='white')
        ax1.scatter(self.ycen_, self.xr_, s=1, color='white')
        ax1.scatter(self.ycen_, self.xc_, s=1, color='k')
        ax1.imshow(self.od.T, clim=self.odlim, aspect='auto', cmap='viridis', origin='lower')
        ax1.set(xlim=[self.yall[0], self.yall[-1]], title='OD and radius')
        ax1.set_axis_off()
        # Plot measured and fitted radius
        ax2.plot(self.yall, self.rad, 'k-')
        ax2.scatter(self.ycen_, self.r_, color='red')
        ax2.set(xlim=[self.yall[0], self.yall[-1]])
        # Plot 3 smaple fits
        for i, j in zip([0, 1, 2], [0, self.r_.shape[0] // 2, -1]):
            axc[i].plot(*self.c_[j].xyfitplot, 'k-')
            axc[i].scatter(*self.c_[j].data, color='red', s=1, alpha=0.5)
            axc[i].set_axis_off()
            axc[i].set(title='Cut @ y = {}'.format(self.ycen_[j]))
        # Adjust layout information
        fig.subplots_adjust(hspace=0.1, wspace=-0.1)
        self.fig = fig

    def fitfun_TF(self, x, x0, rad, amp, m=None, b=None):
        y = amp * (1 - ((x - x0) / rad) ** 2) ** (3 / 2)
        y = np.real(y)
        y[np.isnan(y)] = 0
        if m is not None: y += m * x + b
        return y

# Removing OD gradients in cropped image
class ODFix2D:
    def __init__(self, od, cropi, width=20, odlim=(0, 2), plot=False):
        self.prepare(od, cropi, width, odlim)
        self.nanFix()
        self.fit()
        if plot: self.infoplot()

    def prepare(self, od, cropi, width, odlim):
        self.w = width
        self.odlim = odlim
        self.cropi = tuple([slice(x.start - width, x.stop + width, x.step) for x in cropi])
        # Get od and od bg
        self.od_ = od[self.cropi]
        self.odbg = self.od_.copy()
        self.odbg[width:-width, width:-width] = np.nan
        # Generate z = f(x, y), convert grid to 1d
        self.x, self.y = np.meshgrid(np.arange(self.od_.shape[1]), np.arange(self.od_.shape[0]))
        self.z = self.od_[np.isfinite(self.odbg)]
        self.xy = np.array([self.x[np.isfinite(self.odbg)], self.y[np.isfinite(self.odbg)]])

    def nanFix(self):
        # Bad Points
        x, y = np.meshgrid(np.arange(self.od_.shape[1]), np.arange(self.od_.shape[0]))
        self.odx1 = x[np.logical_not(np.isfinite(self.od_))]
        self.ody1 = y[np.logical_not(np.isfinite(self.od_))]
        # Fix OD
        self.od_ = imagedata.fix_od(self.od_, width=5)

    def fit(self):
        from scipy.optimize import curve_fit
        guess = [0, 0, 0]
        self.fitres, self.fiterr = curve_fit(self.fitfun_2DPoly, self.xy, self.z, p0=guess)
        # Plotting items
        self.bg = self.fitfun_2DPoly_2D(self.x, self.y, *self.fitres)
        self.od = self.od_ - self.bg
        self.od = self.od[self.w:-self.w, self.w:-self.w]

    def fitfun_2DPoly(self, xy, b, m1, m2):
        return b + m1 * xy[0] + m2 * xy[1]

    def fitfun_2DPoly_2D(self, x, y, b, m1, m2):
        return b + m1 * x + m2 * y

    def infoplot(self):
        fig = plt.figure(figsize=(8, 5))
        ax = plt.subplot2grid((2, 3), (0, 0), colspan=2)
        ax1 = plt.subplot2grid((2, 3), (0, 2))
        ax2 = plt.subplot2grid((2, 3), (1, 0))
        ax3 = plt.subplot2grid((2, 3), (1, 1))
        ax4 = plt.subplot2grid((2, 3), (1, 2))

        ax.imshow(self.od_.T, clim=self.odlim, cmap='viridis', aspect='auto', origin='lower')
        ax.scatter(self.ody1, self.odx1, color='red', alpha=0.2, marker='.', s=3)
        ax.plot([self.w, self.w, self.od_.shape[0] - self.w, self.od_.shape[0] - self.w, self.w],
                [self.w, self.od_.shape[1] - self.w, self.od_.shape[1] - self.w, self.w, self.w], 'w-')
        ax.set(xlim=[0, self.od_.shape[0]], ylim=[0, self.od_.shape[1]], title='OD: {} points fixed'.format(self.odx1.size))
        ax.set_axis_off()
        ax1.plot(np.nanmean(self.bg[0:self.w, :], axis=0))
        ax1.plot(np.nanmean(self.odbg[0:self.w, :], axis=0), 'r.', markersize=2)
        ax1.set(title='left')
        ax2.plot(np.nanmean(self.bg[:, 0:self.w], axis=1))
        ax2.plot(np.nanmean(self.odbg[:, 0:self.w], axis=1), 'r.', markersize=2)
        ax2.set(title='bottom')
        ax3.plot(np.nanmean(self.bg[:, -self.w:], axis=1))
        ax3.plot(np.nanmean(self.odbg[:, -self.w:], axis=1), 'r.', markersize=2)
        ax3.set(title='top')
        ax4.plot(np.nanmean(self.bg[-self.w:, :], axis=0))
        ax4.plot(np.nanmean(self.odbg[-self.w:, :], axis=0), 'r.', markersize=2)
        ax4.set(title='right')
        self.fig = fig

###################################################################################


# Convert OD to Density
class OD2Density:
    def __init__(self, od, xsec, pixel, sigma, nmax=np.inf, Ncor=1, plot=False, center=None):
        self.prepare(od, xsec, pixel, sigma, nmax, Ncor, center)
        self.extract_density_all()
        self.find_center_TF()
        if plot: self.infoplot()

    def prepare(self, od, xsec, pixel, sigma, nmax, Ncor, center):
        self.od = od.copy()
        self.xsec = xsec
        self.pixel = pixel
        self.sigma = sigma
        self.nmax = nmax
        self.Ncor = Ncor
        self.center = center

    def extract_density_all(self):
        atomNumber = np.nansum(self.od, axis=1) * self.pixel ** 2 / self.sigma
        atomDensity = atomNumber / (self.xsec.area * self.pixel ** 3) * self.Ncor
        self.atomDensity = atomDensity

    def find_center_TF(self):
        use = self.atomDensity < self.nmax
        c = Curve(x=np.arange(self.atomDensity.shape[0])[use], y=self.atomDensity[use])
        guess = [c.x.shape[0] / 2, c.x.shape[0] / 4, np.max(c.y), np.max(c.y) / 10, np.max(c.y) / 100]
        fitres, fiterr = c.fit(functions.fitfun_TF_harmonic, guess, plot=False)
        y = c.y - (functions.fitfun_TF_harmonic(c.x, fitres[0], fitres[1], 0, fitres[3], fitres[4]))
        if self.center is None: self.center = fitres[0]
        self.nz = Curve(x=(c.x - self.center) * self.pixel, y=y, xscale=1e-6)
        guess = [self.pixel, fitres[1]*self.pixel, fitres[2], fitres[3], fitres[4]/self.pixel]
        self.nz.fit(functions.fitfun_TF_harmonic, guess, plot=False)

    def infoplot(self):
        fig, ax1 = plt.subplots(figsize=(4, 3))
        ax1.scatter(self.nz.x * 1e6, self.nz.y,color='red',alpha=0.5,marker='.',s=7)
        ax1.plot(self.nz.xyfitplot[0]*1e6,self.nz.xyfitplot[1],'k-')


###################################################################################


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
