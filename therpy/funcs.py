#### Table of Contents
__all__ = [
        # C1 : Plotting Functions : add imshow, axvline, axhline
            'add_subplot_axes', 'divide_axes',
        # C2 : Special Mathematical Functions  & Classes
            'gaussian', 'gaussian_2d', 'lorentzian', 'erf', 'erf_shifted', 'erf_box',
            'fourier_transform','real_fast_fourier_transform',
        # C3 : Special Physics Related Functions & Classes
            'cst', 'FermiFunction', 'ThomasFermi_harmonic', 'FermiDirac',
            'betamu2n', 'RabiResonance', 'thermal_wavelength', 'ldB',
            'density_ideal', 'density_virial', 'density_unitary', 'density_unitary_hybrid',
        # C4 : Special Experiment Related Functions  & Classes : box_edge functions
            'ThomasFermiCentering', 'FermiDiracFit', 'box_sharpness', 'interp_od',
            'volt2rabi', 'rabi2volt',
        # C5 : Image Analysis Related
            'get_cropi', 'get_roi', 'get_od', 'fix_od', 'get_usable_pixels', 'com',
            'plot_crop', 'Image', 'XSectionHybrid', 'Hybrid_Image', 'atom_num_filter',
        # C6 : Datatype, I/O related functions :
            'getFileList', 'getpath', 'dictio', 'sdict', 'Curve', 'images_from_clipboard',
            'bin_data',
        # C7 : Numerical Functions :
            'numder_poly', 'numder_gaussian_filter', 'subsampleavg', 'subsample2D',
            'binbyx', 'savitzky_golay', 'surface_fit', 'curve_fit',
            'area_partial_ellipse',
        # C8 : GUI Related
            'qTextEditDictIO',
        # C9 : Legacy
            'AbsImage', 'XSectionTop', 'ODFix2D', 'OD2Density', 'hybridEoS_avg',
            ]

import numpy as np
import pandas as pd
import os
import time
import collections
import urllib.request
import numba
import pickle
import IPython.display
from tqdm import tqdm_notebook as tqdm

from . import dbreader
bec1db = dbreader.Tullia(delta=15)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.ndimage
import scipy.fftpack
import scipy.constants
import scipy.misc
import scipy.optimize
import scipy.integrate
import scipy.special
import scipy.interpolate

from . import imageio


################################################################################
################################################################################
# C1 Plotting Functions
################################################################################
'''
Adds a subplot inside a given axes
==================================
'''
def add_subplot_axes(ax, origin=[0.5,0.25], width=0.5, height=0.3, bg='w', remove=False):
    '''
    Adds a subplot inside a given axes
    ==================================
        Inputs -- all dimensions are in fractional sizes relative to the axes
            ax : main axes
            origin : location of the left bottom corner for the added subplot
            width, height : width and height of the subplot

        Inputs Optional
            bg = 'w' : the color for the background of the subplot, can't be transparent, because needs to cover the background
            remove = False : remove the x,y-axis information from the parent axes

        Returns axes object of the subplot
    '''
    fig = ax.figure
    inax_position  = ax.transAxes.transform(origin)
    x, y = fig.transFigure.inverted().transform(inax_position)
    subax = fig.add_axes([x,y,ax.get_position().width*width,ax.get_position().height*height],facecolor=bg)
    if remove: ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    return subax


'''
Divide an axes into two, horizontally or vertically
===================================================
'''
def divide_axes(ax, divider=0.3, direction='vertical', shared=True):
    '''
    Divide an axes into two, horizontally or vertically
    ===================================================
        Inputs :
            ax : main axes
            divider = 0.3 : fractional size for the bottom subplot
            direction = 'vertical' : or 'horizontal' for division of the axes
            shared = True : share the same x or y axis

        Returns (ax1, ax2) two subplot axes objects
    '''
    # Fractional Dimension for new axes  ; [origin_x, origin_y, width, height]
    if direction == 'horizontal':
        dim1 = [divider, 0, 1-divider, 1]
        dim2 = [0, 0, divider, 1]
    else:
        dim1 = [0, divider, 1, 1-divider]
        dim2 = [0, 0, 1, divider]
    # Add axes
    ax1 = add_subplot_axes(ax, origin=dim1[0:2], width=dim1[2], height=dim1[3], bg='w', )
    ax2 = add_subplot_axes(ax, origin=dim2[0:2], width=dim2[2], height=dim2[3], bg='w', remove=True)
    # Adjust ticks for top (right) axes
    if direction == 'horizontal':
        ax1.tick_params(axis='y', which='both', direction='in', labelbottom=False, right=True)
        ax2.tick_params(axis='y', which='both', direction='in', right=True)
        if shared: ax1.get_shared_y_axes().join(ax1, ax2)
    else:
        ax1.tick_params(axis='x', which='both', direction='in', labelbottom=False, top=True)
        ax2.tick_params(axis='x', which='both', direction='in', top=True)
        if shared: ax1.get_shared_x_axes().join(ax1, ax2)
    # return
    return (ax1, ax2)



################################################################################
################################################################################
# C2 : Special Mathematical Functions  & Classes
################################################################################

'''
gaussian 1D
===========
'''
def gaussian(x, x0=0., sigma=1., amp=1., offset=0., gradient=0., curvature=0.):
    '''
    Gaussian function in 1D
        inputs : (x, x0=0., sigma=1., amp=1., offset=0., gradient=0., curvature=0.)
        output : amp * np.exp( - (x-x0)**2 / (2*sigma**2) ) + offset + gradient * x + curvature * x**2
    '''
    return amp * np.exp( - (x-x0)**2 / (2*sigma**2) ) + offset + gradient * x + curvature * x**2

'''
gaussian 2D
===========
'''
def gaussian_2d(x, y, amp=1., cenx=0., ceny=0., sx=1., sy=1., angle=0., offset=0.):
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

'''
lorentzian
==========
'''
def lorentzian(x, x0=0., gamma=1., amp=1., offset=0., gradient=0.):
    '''
    lorentzian function : (x, x0=0., gamma=1., amp=1., offset=0., gradient=0.)
    gamma = FWHM : Full Width at Half Maximum
    amp is the amplitude at the center
    This function is not normalized to 1
    Area under this Lorentzian is (A * Gamma * pi / 2) with offset and gradient = 0.
    '''
    return amp * ((gamma/2)**2) / ((x-x0)**2 + (gamma/2)**2) + offset + gradient * x

'''
Error Function
==============
'''
def erf(x, x0=0., sigma=1., amp=1., offset=0., gradient=0.):
    '''
    Standard error function :
        sigma defined from gaussian sigma as one standard deviation
        goes from -amp to amp
    To mirror error function, x, x0 => -x, -x0
    '''
    return amp * scipy.special.erf( (x-x0) / (2**0.5 * sigma)) + offset + gradient * x

'''
Shifted Error Function
======================
'''
def erf_shifted(x, x0=0., sigma=1., amp=1., offset=0., gradient=0.):
    '''
    Shifted Error function, such that it starts from 0 and reaches maximum of amp.
    Sigma defined from gaussian sigma as one standard deviation
    To mirror error function, x, x0 => -x, -x0
    '''
    return amp * (erf(x=x, x0=x0, sigma=sigma, amp=0.5, offset=0, gradient=0) + 0.5) + offset + gradient * x

'''
Error Function Box
==================
'''
def erf_box(x, x1=0., x2=0., s1=1., s2=1., amp=1., offset=0., gradient=0.):
    '''Mirrored error functions to create a box '''
    return amp * (erf(x, x0=x1, sigma=s1, amp=0.5) + erf(-x, x0=-x2, sigma=s2, amp=0.5)) + offset + gradient * x

'''
Fourier Transform
'''
def fourier_transform(x, y, k=None):
    '''
    Fourier Transform : f(k) = int_dx {f(x)*exp(-ikx)}
    Normalization : Multiply by 2 / L for proper normalization. Except k=0, which needs 1/L
    inputs (x, y, k=None) returns (k, f(k))
    default k = 100 points from k = 0 to max_k/2
    '''
    if k is None: k = 2*np.pi * np.fft.rfftfreq(y.shape[0], np.diff(x)[0]) / 2
    return [k, np.array([np.trapz(y * np.exp(-1j * ki * x) , x) for ki in k])]

def real_fast_fourier_transform(x, y):
    '''
    Fast Fourier Transform with Real valued f(x)
    inputs (x, y) and returns (k, f(k))
    '''
    return (2*np.pi * np.fft.rfftfreq(y.shape[0], np.diff(x).min()), np.fft.rfft(y, norm='ortho') / np.pi)

################################################################################
################################################################################
# C3 : Special Physics Related Functions & Classes
################################################################################
'''
Constants : Natural & Experimental
==================================
'''
class cst:
    '''
    Constants : Natural & Experimental
    ==================================
        atom = 'Li', 'LiD2', 'LiD1', 'Na', 'NaD1', 'NaD2'  [Na and NaD2 are same, default: LiD2]
    '''
    def __init__(self, **kwargs):
        # Store passed in arguments in self.var
        self.var = kwargs
        # Fill in phsical constants
        self._fill_physical_constants()
        # Find the correct atom type and load its parameters
        if self.var.get('atom', 'Li') is 'Li':
            self.LiD2()
        elif self.var.get('atom') is 'LiD2':
            self.LiD2()
        elif self.var.get('atom') is 'LiD1':
            self.LiD1()
        elif self.var.get('atom') is 'Na':
            self.NaD2()
        elif self.var.get('atom') is 'NaD2':
            self.NaD2()
        elif self.var.get('atom') is 'NaD1':
            self.NaD1()
        else:
            pass  # raise error invalid atom type
        # Fill in the rest
        self._fill_atomic_constants()
        self._fill_exp_constants(**kwargs)
        self._prep_vectorized_interactions()

    # fill in atom specific params
    def LiD2(self):
        self.f = 446.799677e12
        self.tau = 27.102e-9
        self.mass = 9.988346 * 10 ** -27
        self.Isat = 25.4 # W/m^2
        self.atomtype = 'Lithium 6, D2 Line'

    def LiD1(self):
        self.f = 446.789634e12
        self.tau = 27.102e-9
        self.mass = 9.988346 * 10 ** -27
        self.Isat = 75.9 # W/m^2
        self.atomtype = 'Lithium 6, D1 Line'

    def NaD2(self):
        self.f = 508.8487162 * 10 ** 12
        self.tau = 16.249 * 10 ** -9
        self.mass = 3.81754023 * 10 ** -26
        self.atomtype = 'Sodium 23, D2 Line'

    def NaD1(self):
        self.f = 508.3324657 * 10 ** 12
        self.tau = 16.299 * 10 ** -9
        self.mass = 3.81754023 * 10 ** -26
        self.atomtype = 'Sodium 23, D1 Line'

    def _fill_physical_constants(self):
        # From scipy
        self.h = scipy.constants.h
        self.hbar = scipy.constants.hbar
        self.pi = scipy.constants.pi
        self.c = scipy.constants.c
        self.mu_0 = scipy.constants.mu_0
        self.epsilon_0 = scipy.constants.epsilon_0
        self.G = scipy.constants.G
        self.g = scipy.constants.g
        self.e = scipy.constants.e
        self.R = scipy.constants.R
        self.alpha = scipy.constants.alpha
        self.N_A = scipy.constants.N_A
        self.kB = scipy.constants.k
        self.Rydberg = scipy.constants.Rydberg
        self.m_e = scipy.constants.m_e
        self.m_p = scipy.constants.m_p
        self.m_n = scipy.constants.m_n
        # User defined
        self.mu_B = 9.27400899 * 10 ** -24
        self.a_0 = 0.5291772083 * 10 ** -10
        self.twopi = 2 * self.pi

    def _fill_atomic_constants(self):
        self.omega = self.twopi * self.f
        self.wavelength = self.c / self.f
        self.k = self.twopi / self.wavelength
        self.gamma = 1 / self.tau
        self.recoilv = self.hbar * self.k / self.mass
        self.sigma0 = 6 * self.pi * (self.wavelength / self.twopi) ** 2

    ## NEEDS WORK
    # define sets of experimental constants with keywords (ex. LiTopMay2016)
    # keyword using='LiTopMay2016'
    def _fill_exp_constants(self, **kwargs):
        if self.var.get('using', 'Defaults') == 'Defaults':
            self._fill_exp_constants_Defaults()
        elif self.var.get('using') == 'May16Top':
            self._fill_exp_constants_May16Top()
        for key in kwargs.keys(): self.var[key] = kwargs[key]  # Replace ones provided from kwargs

    def _fill_exp_constants_Defaults(self):
        self.sigma = self.sigma0 * self.var.get('sigmaf', 1)
        self.Nsat = self.var.get('Nsat', np.inf)
        self.pixel = self.var.get('pixel', 1)
        self.Ncor = self.var.get('Ncor', 1)
        self.ODf = self.var.get('ODf', 1)
        self.trapw = self.var.get('trapw', 1)
        self.radius = self.var.get('radius', 1)
        self.width = self.var.get('width', 1)
        self.volume = self.var.get('volume', 1)

    def _fill_exp_constants_May16Top(self):
        self.sigma = self.sigma0 * self.var.get('sigmaf', 0.5)
        self.Nsat = self.var.get('Nsat', 120)
        self.pixel = self.var.get('pixel', 1.39e-6)
        self.Ncor = self.var.get('Ncor', 1)
        self.ODf = self.var.get('ODf', 1)
        self.trapw = self.var.get('trapw', 2 * np.pi * 23.9)
        self.radius = self.var.get('radius', 70e-6)
        self.width = self.var.get('width', 80e-6)
        self.volume = self.var.get('volume', np.pi * self.radius ** 2 * self.width)

    def _fill_exp_constants_MonthYearType(self):
        # Must fill out required constants
        pass

    # useful conversions
    def n2kF(self, n, neg=False):
        if neg:
            return (6 * self.pi ** 2 * np.abs(n)) ** (1 / 3) * np.sign(n)
        else:
            return np.real((6 * self.pi ** 2 * n) ** (1 / 3))

    def n2EF(self, n, neg=False):
        return (self.hbar ** 2 * self.n2kF(n, neg) ** 2) / (2 * self.mass) * np.sign(n)

    def EF2n(self, EF, neg=False):
        if neg:
            return 1 / (6 * self.pi ** 2) * (2 * self.mass * np.abs(EF) / self.hbar ** 2) ** (3 / 2) * np.sign(EF)
        else:
            return 1 / (6 * self.pi ** 2) * (2 * self.mass * np.abs(EF) / self.hbar ** 2) ** (3 / 2)

    def kF2EF(self, k):
        return (self.hbar ** 2 * np.real(k) ** 2) / (2 * self.mass)

    def n2EFHz(self, n):
        return self.n2EF(n) / self.h

    def kF2EFHz(self, k):
        return self.kF2EF(k) / self.h

    def EFHz2kF(self, EFHz):
        return self.n2kF(self.EF2n(self.h*EFHz))

    def kT2lambdaDB(self, kT, inHz = False):
        if inHz: kT *= self.h
        return self.h / (self.twopi * self.mass * kT)**(1/2)

    ## Li6 scattering lengths

    def Li6_scattering_length(self, states='13', B=670):
        """ Get the scattering length of a Li6 spin mixture
        Li6_scattering_length(states='13', B), in units of a_0
        """
        if states=='12':
            return a_12_interp(B)
        elif states=='13':
            return a_13_interp(B)
        elif states=='23':
            return a_23_interp(B)

    def n2interaction_strength_one(self, n, states='13', B=670, interp=True):
        """ Get the dimensionless interaction strength for a spin mixture
        n21kFa(n, states, B)
        Needs extrapolation near unitarity since a diverges
        """
        noninterp = 1/(self.n2kF(n) * self.Li6_scattering_length(states, B)*self.a_0)
        if interp:
            # set the extrapolation ranges
            if states=='13':
                above = [682, 687]
                below = [692, 697]
            elif states=='12':
                above = [800, 805]
                below = [812, 817]
            elif states=='23':
                above = [822, 828]
                below = [843, 848]
            Btest = np.hstack([np.linspace(*above, 100),np.linspace(*below, 100)])
            c_ = Curve(Btest, 1/(self.n2kF(n) * self.Li6_scattering_length(states, Btest)*self.a_0))
            def fitfun(x, a0, a1): return a0 + a1*x
            fr, _ = c_.fit(fitfun, [0,-0.1])
            if (B>above[1]) & (B<below[0]):
                return fitfun(B, *fr)
            else:
                return noninterp
        else:
            return noninterp


    def interaction_strength2EFHz(self, interaction, states='13', B=670):
        """ Given a dimensionless interaction strength and B, find the EF needed in the box
        """
        return self.kF2EFHz(1/(interaction*self.Li6_scattering_length(states, B)*self.a_0))

    def BEF2interaction_strength_one(self, B=670, EFHz=1e4, states='13', interp=True):
        """
        Needs extrapolation near unitarity since a diverges
        """
        noninterp = 1/(self.EFHz2kF(EFHz)* self.Li6_scattering_length(states, B)*self.a_0)
        if interp:
            # set the extrapolation ranges
            if states=='13':
                above = [682, 687]
                below = [692, 697]
            elif states=='23':
                above = [800, 805]
                below = [812, 817]
            elif states=='12':
                above = [822, 828]
                below = [843, 848]
            Btest = np.hstack([np.linspace(*above, 100),np.linspace(*below, 100)])
            c_ = Curve(Btest, 1/(self.EFHz2kF(EFHz)* self.Li6_scattering_length(states, Btest)*self.a_0))
            def fitfun(x, a0, a1): return a0 + a1*x
            fr, _ = c_.fit(fitfun, [0,-0.1])
            if (B>above[1]) & (B<below[0]):
                return fitfun(B, *fr)
            else:
                return noninterp
        else:
            return noninterp

    def interaction_strengthEF2B(self,interaction=1, EFHz=1e4, states='13'):
        ''' Fails currently'''
        def funSolve(B): return B - self.BEF2interaction_strength(B=B, EFHz=EFHz,states=states)
        return scipy.optimize.brentq(funSolve, 500,990)

    def _prep_vectorized_interactions(self):
        self.BEF2interaction_strength = np.vectorize(self.BEF2interaction_strength_one)
        self.n2interaction_strength = np.vectorize(self.n2interaction_strength_one)


    ## Very specific conversions
    # Fermi-Surface conversions
    def z2k(self, z):
        return z * (self.mass * self.trapw / self.hbar)

    def N2f(self, N):
        return N * self.h / (self.volume * self.mass * self.trapw)

    def df1D2f(self, df1D):
        return df1D * (-4 * self.pi)

'''Helper FermiFunction : Exact Integral'''
def FermiFunctionExact_(m, z): return scipy.integrate.quad(FermiFunction_Integrand_, 0, np.inf, args=(m, z,))[0] / scipy.special.gamma(m)
def FermiFunction_Integrand_(x, m, z): return (z * np.exp(-x) * x**(m-1)) / (1 + z * np.exp(-x))

'''Helper FermiFunction : Small Z Expansion'''
def FermiFunctionSmallZ_(m, z):
    '''
    Details about the expansion are in Mehran Kardar's notes at
    http://ocw.mit.edu/courses/physics/8-333-statistical-mechanics-i-statistical-mechanics-of-particles-fall-2013/lecture-notes/MIT8_333F13_Lec23.pdf
    '''
    return z - z**2/2**m + z**3/3**m - z**4/4**m + z**5/5**m - z**6/6**m

'''Helper FermiFunction : Large Z Expansion'''
def FermiFunctionLargeZ_(m, logz):
    '''
    Details about the coefficients are in Mathematic file + Mehran Kardar's notes at
    http://ocw.mit.edu/courses/physics/8-333-statistical-mechanics-i-statistical-mechanics-of-particles-fall-2013/lecture-notes/MIT8_333F13_Lec24.pdf
    '''
    return logz**m / scipy.misc.factorial(m) * (1 +
        1.644934067 / logz ** 2 * m * (m - 1) +
        1.894065659 / logz ** 4 * m * (m - 1) * (m - 2) * (m - 3) +
        1.971102183 / logz ** 6 * m * (m - 1) * (m - 2) * (m - 3) * (m - 4) * (m - 5) +
        1.992466004 / logz ** 8 * m * (m - 1) * (m - 2) * (m - 3) * (m - 4) * (m - 5) * (m - 6) * (m - 7) +
        1.998079015 / logz ** 10 * m * (m - 1) * (m - 2) * (m - 3) * (m - 4) * (m - 5) * (m - 6) * (m - 7) * (m - 8) * (m - 9)
    )

'''
FermiFunction PolyLogs
======================
'''
@np.vectorize
def FermiFunction(m, logz):
    '''
    FermiFunction PolyLogs
    ======================
        Inputs:
            m: order of fermi function. Pressure is 5/2, Density is 3/2, etc.
            logz: logz = betamu

        Returns:
            FermiFunction[m, z] = -PolyLog(m, -z)

        Notes for logz input:
            For high z (logz > 10):
                Sommerfeld expansion in log(z) is used with first 6 terms in the sum
            For low z (z < 0.1):
                Summation expansion is used with first 6 terms
            For other values of z:
                The integral is calculated numerically

            These approximation will result in error of at max 1 ppm for m = 1/2, 3/2, 5/2, 7/2.
            Speed improvements are at least 100x.
            At the extreme values, the integral may become unstable, but the approximations are well defined.
    '''
    ZSMALL = 0.1
    LOGZLARGE = 10
    if (logz > LOGZLARGE): return FermiFunctionLargeZ_(m, logz) # Large z limit
    z = np.exp(logz)
    if (z < ZSMALL): return FermiFunctionSmallZ_(m, z) # Small z limit
    return FermiFunctionExact_(m, z) # Exact

'''
Thomas Fermi n(z) for Harmonic Potential
========================================
'''
def ThomasFermi_harmonic(x, x0=0., size=1., amp=0., offset=0., gradient=0.):
    '''
    Thomas Fermi n(z) for Harmonic Potential
    ========================================
        inputs : (x, x0=0., size=1., amp=0., offset=0., gradient=0.)
    '''
    return (np.nan_to_num(np.real(amp * (1 - ((x - x0) / size) ** 2) ** (3 / 2))) + offset + gradient * x)

'''
Fermi Dirac Momentum Distribution
=================================
'''
def FermiDirac(beta, mu, k):
    '''
    Fermi Dirac Momentum Distribution
    =================================
        inputs : (beta, mu, k)
    '''
    return 1 / (1 + np.exp(beta * ((((1.0545718001391127e-34) ** 2 * k ** 2) / (2 * 9.988346 * 10 ** -27)) - mu)))

'''
beta, mu ==> density for Ideal Fermi Gas
========================================
'''
def betamu2n(beta, mu):
    '''
    beta, mu ==> density for Ideal Fermi Gas
    ========================================
        inputs : (beta, mu)
    '''
    # Find k_max when f(k_max) < 1e-3
    k_max, fk_max = 1, FermiDirac(beta, mu, 1)
    while fk_max > 1e-3: k_max, fk_max = k_max * 10, FermiDirac(beta, mu, k_max * 10)
    dk = k_max / 1e5
    # Integrate
    k = np.arange(0, k_max, dk)
    fk = FermiDirac(beta, mu, k)
    return 1 / (2 * np.pi ** 2) * np.sum(k * k * fk) * dk

'''
Rabi oscillations
=================
'''
def RabiResonance(**kwargs):
    '''
    Rabi oscillations
    =================
        Inputs: all kwargs
            pulse duration 		in seconds (t or tau)
            rabi frequency      in Hz (fr or fR) or in s^-1 (wR or wr)
            detuning			in Hz (delta or d or df)
            resonance frequency in Hz (f0) or in s^-1 (w0)
            applied frequency   in Hz (f) or in s^-1 (w)
            --- Combinations ---
            t, fR, delta
            t, fR, f0, f

        Output:
            amplitude * sin^2(Omega_R * tau / 2)
                Omega_R^2 = w_R^2 + delta^2
                amplitude = w_R^2 / Omera_R^2
    '''

    # Extract inputs -- tau, omega, omega0, omegaRabi
    keys = kwargs.keys()
    # tau
    if 'tau' in keys:
        tau = kwargs['tau']
    elif 't' in keys:
        tau = kwargs['t']
    else:
        print("ERROR: pulse duration must be provided using keys 't' or 'tau'")
    # omega rabi
    if 'wR' in keys:
        omega_Rabi = kwargs['wR']
    elif 'fR' in keys:
        omega_Rabi = kwargs['fR'] * cst.twopi
    elif 'wr' in keys:
        omega_Rabi = kwargs['wr']
    elif 'fr' in keys:
        omega_Rabi = kwargs['fr'] * cst.twopi
    else:
        print("ERROR: rabi frequency must be provided using keys 'fr' 'fR' 'wr' 'wR'")
    # delta
    if 'delta' in keys:
        delta = kwargs['delta'] * cst.twopi
    elif 'd' in keys:
        delta = kwargs['d'] * cst.twopi
    elif 'df' in keys:
        delta = kwargs['df'] * cst.twopi
    else:  # delta is not provided
        # resonance frequency
        if 'w0' in keys:
            omega_0 = kwargs['w0']
        elif 'f0' in keys:
            omega_0 = kwargs['f0'] * cst.twopi
        else:
            print("ERROR: resonance frequency must be provided using keys 'f0' 'w0' since delta wasn't provided")
        # applied frequency
        if 'w' in keys:
            delta = kwargs['w'] - omega_0
        elif 'f' in keys:
            delta = kwargs['f'] * cst.twopi - omega_0
        else:
            print("ERROR: rabi frequency must be provided using keys 'fr' 'fR' 'wr' 'wR'")
    # Compute and return
    omega_eff_2 = delta ** 2 + omega_Rabi ** 2
    return omega_Rabi ** 2 / omega_eff_2 * np.sin(np.sqrt(omega_eff_2) * tau / 2) ** 2

'''
deBroglie Wavelength
====================
'''
def thermal_wavelength(kT): return cst_.ldB_prefactor / (kT)**(1/2)
def ldB(kT): return thermal_wavelength(kT)

'''
Fermi Gas Density
=================
'''
@np.vectorize
def density_ideal(kT, mu):
    if kT == 0:
        if mu <= 0:
            print('Density is undefined for negative mu and zero temperature')
            return 0
        return cst_.ideal_gas_density_prefactor * (mu)**(3/2)
    return thermal_wavelength(kT)**(-3) * FermiFunction(m=3/2, logz=mu/kT)
@np.vectorize
def density_virial(kT, mu):
    if kT == 0: return 0
    return kT / thermal_wavelength(kT)**3 * (cst_.virial_coef[0]*1/kT*np.exp(1*mu/kT) + cst_.virial_coef[1]*2/kT*np.exp(2*mu/kT) + cst_.virial_coef[2]*3/kT*np.exp(3*mu/kT) + cst_.virial_coef[3]*4/kT*np.exp(4*mu/kT))
@np.vectorize
def density_unitary(kT, mu):
    if kT < 0: return 0
    if kT == 0: return cst_.EF2n(mu/cst_.xi, neg=True)
    if mu/kT > 4: return cst_.EF2n(mu / precompiled_data_EoS_Density_Generator[0](kT/mu), neg=True)
    if mu/kT > -0.5: return cst_.EF2n(mu / precompiled_data_EoS_Density_Generator[1](mu/kT), neg=True)
    return density_virial(kT, mu)

def density_unitary_hybrid(z, kT_kHz=0, mu0_kHz=1, trap_f=23.9, z0=0, fudge=1, offset=0, gradient=0):
    trap_w = twopi * trap_f
    mu = (mu0_kHz * kHz) - (1/2 * cst_LiD2.mass * (trap_w**2) * (z-z0)**2)
    return density_unitary(kT_kHz * kHz, mu) * fudge + offset + gradient * (z-z0)


################################################################################
################################################################################
# C4 : Special Experiment Related Functions  & Classes
################################################################################

'''
Center n(pixel) using Thomas Fermi Profile
==========================================
'''
def ThomasFermiCentering(*args, **kwargs):
    '''
    Center n(pixel) using Thomas Fermi Profile
    ==========================================
        input : (*args, guess=None, plot=False, output=False, )
    '''
    print("Planned Removal of this Function.")
    # Input parser
    guess = kwargs.get('guess', None)
    plot = kwargs.get('plot', False)
    output = kwargs.get('output', False)
    if len(args) == 1:
        y = args[0].copy()
        x = np.arange(y.size)
    elif len(args) == 2:
        x = args[0].copy()
        y = args[1].copy()
    else:
        print('ERROR: You must provide y alone or x and y. Rest of the imputs must be in key=value format.')
    # Modify, Rescale
    max_y = np.max(y)
    center_x = x[np.equal(y, max_y)][0]
    ym = y / max_y
    xm = x.copy()
    # Guess
    guess = [center_x, x.size / 2, 1, 0]  # r0, rmax, amp, offset
    # Fiting
    res, _ = scipy.optimize.curve_fit(ThomasFermi_harmonic, xm, ym, guess)
    # New x and y
    xn = xm - res[0]
    yn = (ym - res[3]) * max_y
    # Plot and output
    if plot:
        if max_y > 1e5: print('Thomas Fermi fit gave center {:.1f}, radius {:.1f}, amplitude {:.1e}, and offset {:.1e}'.format(res[0], res[1], res[2] * max_y, res[3] * max_y))
        else: print('Thomas Fermi fit gave center {:.1f}, radius {:.1f}, amplitude {:.1f}, and offset {:.1f}'.format(res[0], res[1], res[2] * max_y, res[3] * max_y))
        fig, axes = plt.subplots(figsize=(8, 3), ncols=2)
        axes[0].plot(x, y, x, ThomasFermi_harmonic(x, *res) * max_y)
        axes[0].set(title='Original Data and Fit')
        axes[1].plot(xn, yn, xn, ThomasFermi_harmonic(x, res[0], res[1], res[2] * max_y, 0))
        axes1_2 = axes[1].twinx()
        axes1_2.plot(xn, yn - ThomasFermi_harmonic(x, res[0], res[1], res[2] * max_y, 0), 'r')
        axes[1].set(title='Centered and offseted with error')
    # Outp
    if output:
        outp = {'info': 'ThomasFermiCentering Function did its magic'}
        outp['fitres'] = res
        outp['center'], outp['radius'] = res[0], res[1]
        outp['amp'], outp['offset'] = res[2] * max_y, res[3] * max_y
        if plot: outp['fig'] = fig
        return (xn, yn, outp)
    else:
        return (xn, yn)

'''
Fit Fermi Dirac dist. to momentum distribution
==============================================
'''
def FermiDiracFit(k, fk, **kwargs):
    '''
    Fit Fermi Dirac dist. to momentum distribution
    ==============================================
        inputs : (k, fk, plot=False, cst=cst(), kF={half of max k}, )
    '''
    # Input parser
    plot = kwargs.get('plot', False)
    cst = kwargs.get('cst', cst())
    kF_atoms = kwargs.get('kF', k[k.size // 2])
    EF_atoms = cst.kF2EF(kF_atoms)
    TF_atoms = EF_atoms / cst.kB
    # Prepare variables
    usepts = np.isfinite(fk)
    km, fkm = k[usepts], fk[usepts]
    # Rescale k
    km = km / kF_atoms

    # Fit function
    def FDFit(k, T, mu, amp):
        return amp / (1 + np.exp(1 / T * (k ** 2 - mu)))

    # Guess
    guess = [0.2, 0.9, 1]
    # Fit
    res, _ = scipy.optimize.curve_fit(FDFit, km, fkm, guess)
    # Useful quantities
    outp = {'info': 'Howdy! from FermiDiracFit function'}
    outp['T'] = res[0] * EF_atoms / cst.kB
    outp['beta'] = 1 / (outp['T'] * cst.kB)
    outp['mu'] = res[1] * EF_atoms
    outp['n'] = betamu2n(outp['beta'], outp['mu'])
    outp['kF'] = cst.n2kF(outp['n'])
    outp['EF'] = cst.n2EF(outp['n'])
    outp['EFHz'] = outp['EF'] / cst.h
    outp['TF'] = outp['EF'] / cst.kB
    outp['TTF'] = outp['T'] / outp['TF']
    outp['muEF'] = outp['mu'] / outp['EF']
    x = np.linspace(0, 2, 1000)
    y = FermiDirac(outp['beta'], outp['mu'], x * outp['kF']) * res[2]
    outp['plotdata'] = (x, y)
    outp['disp'] = 'Fermi-Dirac fit results: T = {:.2f} TF; kF = {:2f} M m^-1; EF = {:.2f} kHz; mu = {:.2f} EF'.format(
        outp['TTF'], outp['kF'] * 1e-6, outp['EFHz'] * 1e-3, outp['muEF'])
    # plot
    if plot:
        print(outp['disp'])
        fig, axes = plt.subplots(figsize=(4, 4))
        axes.plot(k / outp['kF'], fk, 'r.')
        axes.plot(x, y)
        outp['fig'] = fig
    return outp

'''
Box Sharpness Calculator -- needs some work
========================
'''
def box_sharpness(data, **kwargs):
    '''
    Box Sharpness Calculator
    ========================
        inputs : (data, **kwargs)
        plot=True
        thickness = 10: number of pixels to average over
        length = 200
        center = com(data)
        using = constants to use :: need to fix this part
        guess_side = None
        guess_top = None
        threshold = np.inf
    '''
    # Treating data
    data[np.logical_not(np.isfinite(data))] = 0
    # Inputs
    plot = kwargs.get('plot', True)
    thickness = kwargs.get('thickness', 10)
    length = kwargs.get('length', 200)
    center = kwargs.get('center', com(data))
    using = kwargs.get('using', 'May16')
    guess_side = kwargs.get('guess_side', None)
    guess_top = kwargs.get('guess_top', None)
    threshold = kwargs.get('threshold', np.inf)
    # Default values
    if using == 'May16':
        thickness = 10
        length = 200
        guess_side = (0.1, 55, length / 2)
        guess_top = (2.5, length / 2 - 26, length / 2 + 26, 3, 3)
    # Get crop regions
    crop1 = get_cropi(data, center=center, width=length, height=thickness)
    crop2 = get_cropi(data, center=center, width=thickness, height=length)
    crop3 = get_cropi(data, center=center, width=length, height=length)
    # Cuts
    y1o = np.sum(data[crop1], axis=0)
    x1o = np.arange(y1o.size)
    y2o = np.sum(data[crop2], axis=1)
    x2o = np.arange(y2o.size)
    # Thresholds
    x1 = x1o[np.less(y1o, threshold)]
    y1 = y1o[np.less(y1o, threshold)]
    x2 = x2o[np.less(y2o, threshold)]
    y2 = y2o[np.less(y2o, threshold)]

    # Fitting functions
    def side_circle(x, amp, rad, x0):
        y = amp * np.real(np.sqrt(rad ** 2 - (x - x0) ** 2))
        y[np.isnan(y)] = 0
        return y

    def top_erf(x, amp, x1, x2, s1, s2):
        return amp * (scipy.special.erf((x - x1) / (np.sqrt(2) * s1)) + scipy.special.erf(-(x - x2) / (np.sqrt(2) * s2)))

    # Fits
    res1, _ = scipy.optimize.curve_fit(side_circle, x1, y1, p0=guess_side)
    res2, _ = scipy.optimize.curve_fit(top_erf, x2, y2, p0=guess_top)
    # Prepare outputs
    outp = {'res1': res1, 'res2': res2, 'thickness': res2[2] - res2[1], 'radius': res1[1]}
    outp['center'] = center
    outp['guess_side'] = guess_side
    outp['guess_top'] = guess_top
    # Plots
    if plot:
        fig, axes = plt.subplots(ncols=3, figsize=(12, 3))
        axes[0].plot(x1o, y1o, x1, y1, x1o, side_circle(x1o, *res1))
        axes[1].plot(x2o, y2o, x2, y2, x2o, top_erf(x2o, *res2))
        axes[2].imshow(data)
        axes[2].scatter(*center)
        axes[2].add_patch(
            patches.Rectangle((center[0] - length / 2, center[1] - length / 2), length, length, fill=False))
        outp['plot'] = fig
    # return
    return outp

'''Helper Function for interp_od'''
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

'''
interpolate OD
==============
'''
def interp_od(Ii, If, img_time):
    '''
    interpolate OD
    ==============
        inputs : (Ii, If, img_time)
        NOTE: img_time must be present in the lookup table, provide in us
        Currently, allowed img_time are only integers upto 15 us
    '''
    return interp_od_special_jit(Ii, If, precompiled_data_Lookup_Table[img_time-1])

'''
convert voltage to Rabi freq
============================
'''
@np.vectorize
def volt2rabi(volt):
    '''
    convert voltage to Rabi freq in kHz
    '''
    if volt < 0.1 or volt > 5:
        return 0
    volt = np.log10(volt)
    dbm = 1.5863 +0.2211*volt -0.1022*volt**2 -0.1301*volt**3 -0.0862*volt**4 +0.2323*volt**5 +0.1624*volt**6 -0.1552*volt**7 -0.1206*volt**8
    dbm = 10**dbm
    sqrtpwr = (10**((dbm-30)/10))**(1/2)
    return -0.0332 +0.5832*sqrtpwr -0.0167*sqrtpwr**2


'''
convert Rabi freq to Voltage
============================
'''
@np.vectorize
def rabi2volt(rabi):
    '''
    convert Rabi freq in kHz to voltage
    '''
    if rabi <= volt2rabi(0.1) or rabi >= volt2rabi(5):
        print('outside valid range')
        return 0
    def funSolve(v): return rabi - volt2rabi(v)
    return scipy.optimize.brentq(funSolve, 0.1, 5)



################################################################################
################################################################################
# C5 : Image Analysis Related
################################################################################

'''
get slice for cropping 2D array
===============================
'''
def get_cropi(data, center=None, width=None, height=None, point1=None, point2=None, point=None, **kwargs):
    '''
    get slice for cropping 2D array
    ===============================
        inputs : (data, center=None, width=None, height=None, point1=None, point2=None, point=None)
        option 1 : center, width, height
        option 2 : point1, point2
        option 3 : point, width, height
    '''
    # Prepare Output
    x = np.arange(0, data.shape[1])
    y = np.arange(0, data.shape[0])
    [XX, YY] = np.meshgrid(x, y)
    cropi = (slice(None, None), slice(None, None))
    # Option 1 -- center width and height
    if center is not None:
        if width is None and height is None: return cropi
        if width is not None and height is None: height = width
        if width is None and height is not None: width = height
        xmin = max(0, center[0] - int(width / 2.0))
        xmax = min(x[-1], xmin + width)
        ymin = max(0, center[1] - int(height / 2.0))
        ymax = min(y[-1], ymin + height)
    # Option 2 -- point1 and point 2
    elif point1 is not None and point2 is not None:
        xmin = max(min(point1[0], point2[0]), 0)
        xmax = min(max(point1[0], point2[0]), x[-1]) + 1
        ymin = max(min(point1[1], point2[1]), 0)
        ymax = min(max(point1[1], point2[1]), y[-1]) + 1
    # Option 3 -- point and width and height
    elif point is not None:
        if width is None and height is None: return cropi
        if width is not None and height is None: height = width
        if width is None and height is not None: width = height
        xmin = max(point[0], 0)
        ymin = max(point[1], 0)
        xmax = min(point[0] + width, x[-1])
        ymax = min(point[1] + height, y[-1])
    else:
        return cropi
    # Return a np array of true false
    return (slice(ymin, ymax), slice(xmin, xmax))

'''
Get roi from center
===================
'''
def get_roi(data, center=[0,0], width=100, height=100):
    '''
    Get ROI Slice for 2D data with respect to the center pixel
    inputs : (data, center=[0,0], width=100, height=100)
    center is the offset from center pixel
    '''
    center = (data.shape[1]//2 + center[0], data.shape[0]//2 + center[1])
    return get_cropi(data, center=center, width=width, height=height)

'''
compute od_BL from wa and woa image
===================================
'''
def get_od(wa, woa, **kwargs):
    '''
    compute od_BL from wa and woa image
    ===================================
        NEEDS speed improvement -- using numba.jit and vectorize
        inputs : (wa, woa, width=5, rawod=False)
        rawod = True fixes the trouble points using width many nearest neighbors.
    '''
    # Inputs
    width = kwargs.get('width', 5)
    rawod = kwargs.get('rawod', False)
    # Compute od
    with np.errstate(divide='ignore', invalid='ignore'): od = np.log(woa / wa)
    od[np.logical_not(np.isfinite(od))] = np.nan
    if rawod: return od # Return raw if asked
    else: return fix_od(od, width) # return fixed od

'''
fix nan and inf in 2d array using srrounging pixels
===================================================
'''
def fix_od(odIn, width = 5):
    '''
    fix nan and inf in 2d array using srrounging pixels
    ===================================================
        inputs : (odIn, width = 5)
    '''
    # Inputs
    od = odIn.copy()
    # Find trouble points
    X, Y = np.meshgrid(np.arange(od.shape[1]), np.arange(od.shape[0]))
    od[np.logical_not(np.isfinite(od))] = np.nan
    pts = np.logical_not(np.isfinite(od))
    Xp, Yp = X[pts], Y[pts]
    # Average neighbors
    for i, x in enumerate(Xp):
        cropi = get_cropi(X, center=(Xp[i], Yp[i]), width=width)
        replace = od[cropi].flatten()
        replace = replace[np.isfinite(replace)]
        if replace.size == 0:
            replace = 0
        else:
            replace = np.mean(replace)
        od[Yp[i], Xp[i]] = replace
    # return fixed od
    return od

'''
get Bool 2d array of good pixels from wa and woa
================================================
'''
def get_usable_pixels(wa, woa, **kwargs):
    '''
	Inputs: (wa, woa, threshold=25)
		threshold => (wa-dark) must be >= threshold for pixel to be usable; default 25

	Output:
		use
		True only if (wa-dark) >= threshold and rawod is a finite number
	'''
    # Input parser
    threshold = kwargs.get('threshold', 25)
    # Fill in
    rawod_finite = np.isfinite(get_od(wa, woa, rawod=True))
    threshold_condition = np.greater_equal(wa, threshold)
    return np.logical_and(rawod_finite, threshold_condition)

'''
center of mass
==============
'''
def com(data):
    '''
    center of mass
    ==============
        inputs : data
        return (com_x, com_y)
    '''
    X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    total = np.sum(data)
    x0 = np.sum(data * X) / total
    y0 = np.sum(data * Y) / total
    return (x0, y0)

'''
Plot crop rectangle on 2D data
==============================
'''
def plot_crop(data, *args, **kwargs):
    '''
    Plot crop rectangle on 2D data
    ==============================
        inputs : (data, *args, **kwargs)
        option 1 : data, cropi
        option 2 : data, center = [42,42], width = 42, height = 42
    '''
    # Get cropi
    if len(args) > 0:
        if type(args[0]) is slice:
            cropi = args[0]
            center = None
        else:
            cropi = get_cropi(data, **kwargs)
            center = kwargs.get('center', None)
    # Plot
    fig, axes = plt.subplots(figsize=(4, 4))
    axes.imshow(data)
    if center is not None: axes.scatter(*center)
    axes.add_patch

'''
Image Class
===========
'''
class Image:
    '''Get image name and path, start self.var'''
    def __init__(self, name=None, path=None, od=None, **kwargs):
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

    def __str__(self): return 'Image object'
    def __repr__(self): return 'Image object'

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
    def sigma(self,): return self.var.get('sigma', cst_.sigma0 * self.sigmaf)

    @property
    def memory_saver(self,): return self.var.get('memory_saver')

    @property
    def lookup_table_version(self,): return self.var.get('lookup_table_version')


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
                cropi = get_cropi(Ii, **self.cropset)  # Need to improve speed here, takes 50 ms, (99% of time spent at [XX, YY] = np.meshgrid(x, y))
                Ii = Ii[cropi]
                If = If[cropi]
            elif (task == 'rotate') and (self.rotate != 0):
                Ii = scipy.misc.imrotate(Ii, angle=self.rotate, interp=self.rotate_method) # Takes 250 ms
                If = scipy.misc.imrotate(If, angle=self.rotate, interp=self.rotate_method) # takes 250 ms
            elif (task == 'subsample') and (self.subsample != 1):
                Ii = subsample2D(Ii, bins=[self.subsample, self.subsample]) # 1 ms
                If = subsample2D(If, bins=[self.subsample, self.subsample]) # 1 ms
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
        elif method in ['sBL']:
            with np.errstate(divide='ignore', invalid='ignore'): self.var['od'] = - np.log(self.sf / self.si) + self.si - self.sf
        else:
            with np.errstate(divide='ignore', invalid='ignore'): self.var['od'] = - np.log(self.sf / self.si)
        self.var['recalc'][2] = False

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

'''
Cross Section Hybrid
====================
'''
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

'''
Hybrid Image
============
'''
class Hybrid_Image(Image):
    def __init__(self, name=None, path=None, od=None, **kwargs):
        # Initialize the complete Image Object
        super(Hybrid_Image, self).__init__(name=name, path=path, od=od, **kwargs)

        Default_Hybrid_Image = dict(ellipticity=1, xsec_extension='default',xsec_slice_width=4,
                                    xsec_fit_range=1.75, xsec_override=False, trap_f=23.9,
                                    radial_selection=0.7, trap_center_override=False, kind='unitary',
                                    Tfit_lim=0.2, Tfit_guess_kT=0.5, Tfit_guess_mu0=1)

        Level_Selector_Hybrid_Image = [['ellipticity','xsec_extension', 'xsec_slice_width','xsec_fit_range', 'xsec_override'],
                                       ['fudge','sigmaf','sigma','pixel','trap_f','radial_selection','trap_center_override'],
                                       ['kind','Tfit_lim', 'Tfit_guess_kT', 'Tfit_guess_mu0']]

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
    def u(self,): return 0.5*cst_LiD2.mass*self.trap_w**2*self.z**2

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
    def nu(self,): return Curve(x=self.u, y=self.n, xscale=1e3*cst_LiD2.h, yscale=1e18)

    @property
    def EFu(self,): return Curve(x=self.u, y=cst_LiD2.n2EF(self.n, neg=True), xscale=1e3*cst_LiD2.h, yscale=1e3*cst_LiD2.h)

    @property
    def ku(self,):
        EFu = self.EFu.sortbyx().subsample(bins=2)
        ku = EFu.diff(method='poly', order=1, points=4)
        return Curve(x=ku.x, y=-ku.y, xscale=ku.xscale).subsample(bins=2)

    @property
    def kz_u(self,):
        ku = self.ku
        z = (2 * ku.x / cst_LiD2.mass / self.trap_w**2 )**(1/2)
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
        return self.T_kHz * 1e3 * cst_LiD2.h / cst_LiD2.n2EF(self.n, neg=True)

    @property
    def TTF_center(self,):
        fitfun, fitres = self.var['Tfit_info']
        n_center = np.max(fitfun(self.nz.x, *fitres))
        return self.T_kHz * 1e3 / cst_LiD2.n2EFHz(n_center)

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
                kT *= 1e3*cst_LiD2.h
                mu0 *= 1e3*cst_LiD2.h
                mu = mu0 - 1/2 * cst_LiD2.mass * self.trap_w**2 * (z-z0*1e-6)**2
                return density_unitary(kT, mu) + a0
        elif (self.kind == 'ideal') or (self.kind == 'polarized'):
            def fitfun(z, kT, mu0, a0=0, z0=0):
                kT *= 1e3*cst_LiD2.h
                mu0 *= 1e3*cst_LiD2.h
                mu = mu0 - 1/2 * cst_LiD2.mass * self.trap_w**2 * (z-z0*1e-6)**2
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

'''
Atom Num Filter for DataFrame
'''
def atom_num_filter(df, keep=0.10, offset=0.0, using='ABS', display=False, plot=False, ax=None):
    '''
    Filter atom number and store df.use bool array in the dataframe
    Inputs :
        df : the datafame where atom numebrs are calculated, output of images_from_clipboard
        keep : fraction of atoms to keep from median value
        offset : offset from median value ==> medial * (1+offset)
        using : NOT YET IMPLEMENTED
        display : print a summery of the number of images removed
        plot : make a plot of atom numbers vs time and histogram of kept atoms
        ax : optionally provide list of two axes to plot into
    Returns the list of ax object
    '''
    # Filter
    df.total_atoms = 0
    for n,r in df[df.download].iterrows():
        df.loc[n, 'total_atoms'] = r.image.total_atoms
    median_numbers = np.median(df[df.download].total_atoms)
    offset += 1
    df['use'] = False
    for n,r in df[df.download].iterrows():
        df.loc[n,'use'] = (r.total_atoms > median_numbers*(offset-keep)) & (r.total_atoms <= median_numbers*(offset+keep))
    fudge = df[df.download].iloc[0].image.fudge
    mean_numbers = np.mean(df[df.use].total_atoms)
    std_numbers = np.std(df[df.use].total_atoms)

    # Plots
    if display:
        print("Total Images {} ==> Good {}, Bad {}".format(np.sum(df.download), np.sum(df.use), np.sum(df.download) - np.sum(df.use)))
    if plot:
        if ax is None: fig, ax = plt.subplots(ncols=2, figsize=[10, 5])
        else: fig = ax[0].figure

        # atomnum vs time plot
        ax[0].plot(df[df.download].time, df[df.download].total_atoms / 1e6, 'C0.-')
        ax[0].plot(df[~df.use].time, df[~df.use].total_atoms / 1e6, 'rx')
        ax[0].axhline(median_numbers/1e6, linestyle='-', c='k', alpha=0.7)
        ax[0].axhspan(median_numbers/1e6 * (1 - keep), median_numbers/1e6 * (1 + keep), color='k', alpha=0.05)
        ax[0].set(xlabel='Time (minutes)', ylabel='Atom Number (million)',
                  title='Fudge {}; Mean {:.3f} million'.format(fudge, mean_numbers/1e6))

        # Histogram plot
        ax[1].hist(df[df.use].total_atoms.values / 1e6, bins=10)
        ax[1].axvline(mean_numbers/1e6, linestyle='-', c='k', alpha=0.7)
        ax[1].set(xlabel='Atom Number (million)', ylabel='Counts',
                  title=r'Atom Num {:.3f} $\pm$ {:.3f} million'.format(mean_numbers/1e6, std_numbers/1e6))

    return ax

################################################################################
################################################################################
# C6 : I/O related functions
################################################################################
'''
Get a list of files in a folder
===============================
'''
def getFileList(folder = 'Not Provided', fmt = '.', outp = False):
    '''
    Get a list of files in a folder
    ===============================
        inputs : (folder, fmt='.', outp=False)
        if outp: return (names, paths)
        else: return filenames
    '''
    # Confirm that given folder path is correct
    if not os.path.exists(folder): raise ValueError("Folder '{}' doesn't exist".format(folder))
    # Verify format
    if type(fmt) is not str: raise ValueError("Format (fmt) must be a string")
    if fmt[0] != '.': fmt = '.' + fmt
    def useFile(filename):
        if fmt == '.': return os.path.splitext(filename)[1] != ''
        return os.path.splitext(filename)[1] == fmt
    # Folder contents
    filenames = [filename for filename in os.listdir(folder) if useFile(filename)]
    # Output
    if outp:
        names = [os.path.splitext(f)[0] for f in filenames]
        paths = [os.path.join(folder,f) for f in filenames]
        return (names, paths)
    return filenames

'''
Get full path to a location relative to My Programs
===================================================
'''
def getpath(*args):
    '''
    Get full path to a location relative to My Programs
    ===================================================
        inputs : (*args)
        can be : single string of relative path, list of sub folders, *args of strings
    '''
    basepath = os.path.expanduser('~')
    rootpath = os.path.join(basepath, 'Documents', 'My Programs')
    # Process input
    if len(args) == 0: return rootpath
    elif len(args) == 1 and type(args[0]) is str: relative = args       # Single string
    elif len(args) == 1 and type(args[0]) is list: relative = args[0]   # A single list
    elif len(args) >= 2 and type(args[1]) is str: relative = args       # Bunch of strings
    else:
        print("Illegal type of relative path received at therpy.getpath")
        return os.path.join(rootpath,'Junk')
    # Extend and return
    return os.path.join(rootpath,*relative)

'''
Needs Work : subclass dict : add store and retrive information from file
=============================
'''
class dictio:
    '''
    Needs Work : subclass dict : add store and retrive information from file
    =============================
    Get inspiration from sdict
    '''
    def __init__(self,*args,**kwargs):
        # Get data from the inputs
        if len(args) is 1 and type(args[0]) is dict: self.data = args[0]
        elif len(args) is 1 and type(args[0]) is str: self.data = self.fromfile(filepath=args[0])
        elif "filepath" in kwargs.keys(): self.data = self.fromfile(filepath=kwargs['filepath'])
        else: self.data = kwargs

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def __iter__(self):
        return iter(self.data.keys())

    def tofile(self,filepath=None):
        current_time_str = time.strftime("%m-%d-%Y_%H_%M_%S_dictioOutput.txt")
        default_dictio_filepath = getpath('Default Storage','dictio',current_time_str)
        if filepath is None: filepath = default_dictio_filepath
        fid = open(filepath,'w')
        for key in self.data.keys():
            val = self.data[key]
            typestr = str(type(val))
            typestr = typestr[typestr.find("'")+1:]
            typestr = typestr[:typestr.find("'")]
            fid.write("{}\t{}\t{}\n".format(key, val, typestr))
        fid.close()
        return filepath

    def fromfile(self,filepath=None):
        data = dict()
        if filepath is None: return data
        fid = open(filepath,'r')
        for line in fid:
            key = line[:line.find("\t")]
            line = line[line.find("\t")+1:]
            val = line[:line.find("\t")]
            line = line[line.find("\t")+1:]
            typestr = line[:line.find("\n")]
            if typestr == 'str': val = str(val)
            elif typestr == 'int': val = int(val)
            elif typestr == 'float': val = float(val)
            data[key] = val
        return data

    def get(self,key,default=None):
        if key in self.data.keys():
            return self.data[key]
        else:
            return default

'''
Smarter Dictionary
==================
'''
class sdict(collections.OrderedDict):
    '''
    Smarter Dictionary : Remembers order, let's you index, and save data
    Initialize :
        Method 1 -- define it without consideration to initial order
            d = sdict(b='B', c='C', a='A', )
        Method 2 -- define it one by one, order is maintained
            d = sdict(); d['b'] = 'B'; d['c'] = 'C'; d['a'] = 'A'
        Method 3 -- define using zip from two lists, order is maintained
            k = ['b','c','a']; v = ['B','C','A']
            d = sdict(zip(k, v))
    Access/Modify Data by Index or Key : d[0] or d['b'] are identical
    New entries MUST be added via Key
    Slicing will work similar to a list of values
    Setting data via slicing will also work similar to a list
    Iterating as list [*d] or as dict {**d}
    Additional Properties : k and v for a list of keys and values
    '''
    def __init__(self, *args, **kwargs):
        super(sdict, self).__init__(*args, **kwargs)
    def __getitem__(self, k):
        if type(k) == slice: return list(self.values())[k]
        if type(k) in [int, float]: k = list(self.keys())[int(k)]
        return super(sdict, self).__getitem__(k)
    def __setitem__(self, k, v):
        if type(k) == slice:
            for ki,vi in zip(list(self.keys())[k], v): super(sdict, self).__setitem__(ki, vi)
        elif type(k) in [int, float]: super(sdict, self).__setitem__(list(self.keys())[int(k)], v)
        else: super(sdict, self).__setitem__(k, v)
    def __iter__(self):
        for v in list(self.values()): yield v
    @property
    def k(self): return list(self.keys())
    @property
    def v(self): return list(self.values())

'''
Curve {x, y} Data
=================
'''
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
        return self.copy(*binbyx(self.x, self.y, **kwargs))

    def subsample(self, bins=2):
        return self.copy(*subsampleavg(self.x, self.y, bins=bins))

    def diff(self, **kwargs):
        method = kwargs.get('method', 'poly')
        if method == 'poly':
            dydx = numder_poly(self.x, self.y, order=kwargs.get('order', 1), points=kwargs.get('points', 1))
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
            try:
                fitres, fiterr = scipy.optimize.curve_fit(fitfun, self.x[using], self.y[using], p0=guess, bounds=bounds)
                fiterr = np.sqrt(np.diag(fiterr))
            except RuntimeError as err:
                fitres = guess
                fiterr = guess
                print("CAN'T FIT, Returning Original Guess: Details of Error {}".format(err))
        else:
            try:
                fitfun_ = lambda p: fitfun(self.x[using], *p) - self.y[using]
                fitres_ = scipy.optimize.least_squares(fun=fitfun_, x0=guess, loss=loss, f_scale=noise, bounds=bounds)
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

'''
images_from_clipboard
=====================
'''
def images_from_clipboard(df=None, x='time', params=[], image_func=Image, download='ABS', display=False, verify=False, keep_all=False):
    '''
    Get a list of images from clipboard and desired experimental parameters
    Inputs
        df = None : if you desire to add images to already polulated dataframe
        x = None : a string specifying the x parameter, default will be set to image time in minutes
        params = [] : a list of string, the names of parameters you want to add to the dataframe, in addition to x
        image_func = tp.Image : the function that takes in imagename string and returns image object, tp.Image or tp.Hybrid_Image
        download = 'ABS' : which images to download ==> 'A', 'B', 'S', 'AB', 'SB', 'ABS', 'SAB'
        display = False : wether to print some information or not
        verify = False : Remove any file that doesn't have its partners. For example if S present, but AB missing, in case uing='ABS'
        keep_all = False : wether to keep images that are not part of download
    '''
    # Generate new empty dataframe if not provided
    if df is None: df = pd.DataFrame()

    # Fix the list of parameters
    required_params = ['name','time','image','A','B','S','download']
    if (x not in params) and (x != 'time'): params = params + [x]
    if 'unixtime' in params: params.remove('unixtime')

    # Prepare DataFrame for inserting new data
    if df.size == 0: df = pd.DataFrame(None, columns = required_params + params).set_index('name')
    else:
        old_columns = df.columns.tolist() + [df.index.name]
        for n in (required_params + params):
            if n not in old_columns: df[n] = None

    # Get the list of filenames from clipboard and add them to DataFrame
    copied_names = pd.read_clipboard(header=None)[0].tolist()
    old_names = df.index.tolist()
    for n in copied_names:
        if n not in old_names: df.loc[n] = None
    df = df.sort_index()

    # Get the parameters
    try: bec1db.refresh()
    except: pass
    df_params = bec1db.image_query(df.index.tolist(), params + ['unixtime'])
    df_params = df_params.rename(columns={'imagename':'name', 'unixtime':'time'}).set_index('name')
    for n in (params + ['time']):
        df[n] = df_params[n]

    # Fix the parameters
    df.time = (df.time - df.time.min())/60
    df.A = [n[-1] == 'A' for n in df.index]
    df.B = [n[-1] == 'B' for n in df.index]
    df.S = [n[-1] not in ['A','B'] for n in df.index]
    df['x'] = df[x]

    # Add Images
    download = list(download)
    df.download = (df.A & ('A' in download)) | (df.B & ('B' in download)) | (df.S & ('S' in download))
    for n in tqdm(df.index.values):
        if df.loc[n,'download'] and (type(df.loc[n,'image']) != image_func): df.loc[n, 'image'] = image_func(n)

    # Remove entries that are not in download
    if not keep_all:
        df = df[df.download]

    # Verify if Required
    if verify:
        remove_x = []
        reqd = ('A' in download) + ('B' in download) + ('S' in download)
        for x in np.unique(df.time):
            if df[(df.time == x) & df.download].shape[0] != reqd: remove_x.append(x)
        for x in remove_x:
            print('Removing ', df[df.time == x].index.values)
            df = df[df.time != x]

    # Display
    if display:
        print('Total Number of Shots {}, Images {}'.format(np.unique(df.time).shape[0],df.shape[0]))
        IPython.display.display(df.head(3))

    return df

'''
bin data
========
'''
class bin_data:
    '''
    Binning data with same x values
    ===============================

    Properties : self.xxxx
        xi, yi
        x : unique values of xi in increasing order
        y : mean value at x
        ynum : number of points at x
        ybin : list of points at x
        ystd : np.std at x
        yerr : ystd/sqrt(ynum), standard error of the mean
        data : (x, y)
        std : (x, y, ystd)
        err : (x, y, yerr)
        all : (xi, yi)
    '''
    def __init__(self, xi, yi, ):
        x = np.unique(xi)
        ybin = [yi[xi == x_i] for x_i in x]
        ynum = np.array([len(y_i) for y_i in ybin])
        y = np.array([np.mean(y_i) for y_i in ybin])
        ystd = np.array([np.std(y_i) for y_i in ybin])
        yerr = ystd / np.sqrt(ynum)

        # Store
        self.xi = xi
        self.yi = yi
        self.x = x
        self.y = y
        self.ybin = ybin
        self.ynum = ynum
        self.ystd = ystd
        self.yerr = yerr
        self.data = (x, y)
        self.std = (x, y, ystd)
        self.err = (x, y, yerr)
        self.all = (xi, yi)


################################################################################
################################################################################
# C7 : Numerical Functions
################################################################################
'''
Numerical Derivative by Polynomial Fits
=======================================
'''
def numder_poly(x,y,order=2,points=1):
    '''
    Numerical Derivative by Polynomial Fits
    =======================================
        inputs : (x, y, order=2, points=1)
        x, y data arrays
        order of fitted Polynomial
        # of points to use on the left and right. Fit happens to 2*points+1 points
    '''
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

'''
Numerical Derivative by Gaussian Filter
=======================================
'''
def numder_gaussian_filter(x,y,**kwargs):
    '''
    Numerical Derivative by Gaussian Filter
    =======================================
        Now Working Yet
        Inputs : (x, y, sigma=2, mode='constant', order=1, )
    '''
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

'''
sub sample average of 1D array
==============================
'''
def subsampleavg(*args,**kwargs):
    '''
    sub sample average of 1D array
    ==============================
        input : (*args, bins=2, )
    '''
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

'''
sub sample sum of 2D array
==========================
'''
def subsample2D(x, bins=(1,1)):
    '''
    sub sample sum of 2D array
    ==========================
        inputs : (x, bins=(2,1))
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

'''Helper Function - binbyx'''
def binbyx_array(*args,**kwargs):
    # Bin y1, y2, ... by x using given bins array center
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

'''
Binning data by x
=================
'''
def binbyx(*args, **kwargs):
    '''
    Binning data by x
    =================
        binbyx(*args, **kwargs)

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

'''
savitzky_golay 1D smoothing
===========================
'''
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

'''
surface fitting
===============
'''
def surface_fit(*args, **kwargs):
    '''
    surface fitting
    ===============
        inputs : (*args, fun, guess, disp=False, output=False, show=False, )
        *args could be (z, ) or (x, y, z) or (z, using, ) or (x, y, z, using)
        x, y, z must have the same shape : x, y must be of meshgrid kind
        output = True for detailed report of the fit, else olyy fit valeus are returned
        disp = True for print from fit routine
        show = True for plot
    '''
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
    res = scipy.optimize.least_squares(fitfunc, guess, verbose=disp)

    # outputs
    res['zfit'] = fun(x, y, *res.x)

    # Show
    if show:
        fig,ax = plt.subplots(nrows=1,ncols=1)
        ax.imshow(z,cmap='gray', origin=0)
        ax.contour(res['zfit'], cmap='jet')
        res['fig'] = fig

    if output: return res
    else: return res.x

'''
curve fitting
=============
'''
class curve_fit:
    '''
    Least square curve fitting to {x,y} data
    ========================================
    Inputs Required :
        fitfun - python function with inputs (of type x_data, p0, p1, p2, p3, ...) and returns np.array y_data
        guess  - could be a simple list, or more detailed dict (see below for more info)
        x_data - I think this could be anything supported by fitfun input, to be safe, keep it np.array
        y_data - must be np.array

    Inputs Optional :
        y_err - sigma for the y_data, defaults to np.ones_like(y_data). The following is optimized sum ( (y_data-fitfun(...))/y_err )^2
        fixed - dict(gradient = 0, tau = 5, ...) : parameters to fix. NOTE: if guess contains the same key, that key will not be fixed
        plot  - True / False : Plots a standard figure with data, fit line, residuals
        info  - True / False : print(self.fr)

    guess :
        1) guess = [12, 3, -np.pi, ...] : a simple list
        2) guess = [ [12, (10, 14), 'meter'], 3, [np.pi, (0, 2*np.pi)], ...] : mixture of simple value and fancy list
        A fancy list for a guess contains up to three items -- the guess value itself, a tuple for lower and upper bounds, a string for units
        A fancy list must contain the guess itself, but the other two things are optional
        3) guess = dict(tau = 12, phase = [np.pi, (0, 2*np.pi), 'radians'], ...) : a dictionary containing mixture of simple value and fancy list

    MUST DO :
        - Make sure x_data, y_data, and y_err is valid for fitfun input/output and DOES NOT contain inf or nan.

    Properties : self.xxxxxx
        x, y, y_err, success, fitfun
        fr : pandas dataframe of the fitresults
        fv : indexable sdict of fit values with parameters as keys, including the fixed and default parameters
        fe : indexable sdict of fit errors with parameters as keys, including the fixed and default parameters
        ul : indexable sdict of upper limit of fit values with parameters as keys, including the fixed and default parameters
        ll : indexable sdict of lower limit of fit values with parameters as keys, including the fixed and default parameters
        xp : finely spaced grid for same x range : np.linspace(self.x.min(), self.x.max(), 1000)
        ye : short form for y_err

    Special Methods :
        __call__ : returns fitted curve at optionally provided x and overridden parameters : self(), self(x, phase=0, amp=0, ...), self(self.xp),
        __len__ : len(self) returns the number of parameters the fitfun takes
        __getitem__ : self['amp'] or self[1] : returns the fir value for given key or index
        __setitem__ : self['amp'] = 0 or self[1] = 0 : fixes the
        __bool__ : if self: or bool(self) : true if the fit was successful
        __repr__ or __str__ : str(self) or print(self) : returns the fitresults in printable format

    Methods : self.xxx(aa=yy, bb=yy, ..)
        yband(x = None, using=[]) :
        plot()
            two plots, data and fit curve on top, and residuals below. Optional inputs
            ax : to plot somewhere specific
            fiterrors = False : include fit error band along with optimal fit
            using = [] : which parameters to sue to calculate fiterrors band
            divider = 0.25 : how to split the axes for residuals
        plot_fitdata(ax=None, x=None) : data and fit curve
        plot_residuals(ax=None) : residuals
        plot_residuals_hist(ax=None, orientation='vertical) : histogram of the residuals
        plot_fiterrors(ax=None, x=None, using=[]) : fit error band, optinally include only keys in using
    '''
    def __init__(self, fitfun, guess, x_data, y_data, y_err=None, fixed=dict(), plot=False, info=False):
        ''' init will fit the data and save fitresults '''
        ### Verify inputs
        if not callable(fitfun): print("provided fitfun is not valid python function!")

        ### Process single item from guess -- return guess value, bounds, units
        def temp_process_guess_item(item):
            units, bounds = None, [-np.inf, np.inf] # default value
            if type(item) in [list, tuple, np.ndarray]:
                value = item[0]
                if len(item) > 3: return [value, bounds, units]
                for i in item[1:]:
                    if type(i) is str: units = i
                    elif type(i) in [list, tuple, np.ndarray]: bounds = i
            else: value = item
            return [value, bounds, units]

        ### Process guess -- generate guess_keys, guess_values, guess_bounds, guess_units
        # the order of keys will be determined by the order of fitfun input order
        if type(guess) == dict:
            guess_keys = [k for k in fitfun.__code__.co_varnames[1:fitfun.__code__.co_argcount] if k in list(guess.keys())]
            temp_ = [temp_process_guess_item(guess[k]) for k in guess_keys]
        elif type(guess) in [list, tuple, np.ndarray]:
            guess_keys = fitfun.__code__.co_varnames[1:1+len(guess)]
            temp_ = [temp_process_guess_item(i) for i in guess]
        elif (type(guess) in [float, int]) or (np.issubdtype(guess, np.number)):
            guess_keys = fitfun.__code__.co_varnames[1:2]
            temp_ = [temp_process_guess_item(guess)]
        else:
            print('Does NOT understand data type of guess : ', guess)
        guess_values, guess_bounds, guess_units = np.array([i[0] for i in temp_]), np.array([i[1] for i in temp_]).T, [i[2] for i in temp_]

        ### Extract all fixed items, including provided and default ones
        fixed_func_defaults = {k:v for k, v in zip(fitfun.__code__.co_varnames[-len(fitfun.__defaults__):fitfun.__code__.co_argcount], fitfun.__defaults__)}
        fixed_dict = {**fixed_func_defaults, **fixed}
        for k in guess_keys: fixed_dict.pop(k, None)

        ### Define temp fitfun for internal use only
        def fitfun_args(x, *args):
            return fitfun(x, **{k: v for k, v in zip(guess_keys, args)}, **fixed_dict)

        ### Fit Data
        success = False
        try:
            fv_, fe_ = scipy.optimize.curve_fit(fitfun_args, x_data, y_data, guess_values, sigma=y_err, bounds=guess_bounds)
            fe_ = np.sqrt(np.diag(fe_))
            success = True
        except (ValueError, RuntimeError) as err:
            fv_, fe_ = guess_values, guess_values*np.nan
            print("CAN'T FIT, Returning Original Guess: Details of Error :: {}".format(err))

        ### Formatting Fit Results
        fitresults_dict = dict(FitValue=fv_, FitError=fe_, Units=guess_units, Guess=guess_values, LowerBound=guess_bounds[0], UpperBound=guess_bounds[1])
        fitresults_df = pd.DataFrame(fitresults_dict, index=guess_keys, columns=['FitValue','FitError','Units','Guess','LowerBound','UpperBound'])
        for k, v in fixed_dict.items(): fitresults_df.loc[k] = [v, 0, None, v, v, v]
        fitresults_df = fitresults_df.loc[fitfun.__code__.co_varnames[1:fitfun.__code__.co_argcount], :] # sort the index by function input list
        fitresults_df['FitError%'] = np.nan_to_num(np.abs(fitresults_df['FitError'] / fitresults_df['FitValue'])) * 100

        ### Store results to self
        self.fr = fitresults_df
        self.fitfun = fitfun
        self.success = success
        self.x = x_data
        self.y = y_data
        self.y_err = y_err

        ### Plots and display
        if plot: self.plot()
        if info: print(self)

    @property
    def fv(self): return sdict(zip(self.fr.index.values, self.fr['FitValue'].values))
    @property
    def fe(self): return sdict(zip(self.fr.index.values, self.fr['FitError'].values))
    @property
    def ul(self): return sdict(zip(self.fr.index.values, self.fr['FitValue'].values + self.fr['FitError'].values))
    @property
    def ll(self): return sdict(zip(self.fr.index.values, self.fr['FitValue'].values - self.fr['FitError'].values))
    @property
    def xp(self): return np.linspace(self.x.min(), self.x.max(), 1000)
    @property
    def ye(self): return self.y_err

    def __call__(self, *args, **kwargs):
        '''get the fit line evaluated at *args (self.x) with **kwargs overridden'''
        x = args[0] if len(args) == 1 else self.x
        return self.fitfun(x, **{**self.fv, **kwargs})
    def __len__(self): return len(self.fv)
    def __getitem__(self, key):
        if (type(key) in [int, float]): return self.fv[int(key)]
        elif type(key) == str: return self.fv.get(key, None)
    def __setitem__(self, key, value):
        if type(key) in [int, float]: key = self.fr.index.values[int(key)]
        r = self.fr.loc[key].values
        r[0], r[1], r[6] = value, 0, 0
        self.fr.loc[key] = r
    def __bool__(self): return self.success
    def __repr__(self): return self.fr.to_string()
    def __str__(self): return self.fr.to_string()

    def yband(self, x=None, using=[]):
        '''Return (y_min, y_max) at x including using list of fit errors'''
        if x is None: x = self.x
        if type(using) == str: using = [using,]
        if len(using) == 0: using = self.fr.index.values
        ys = [self(x)]
        for k in using:
            ys.append(self(x, **{k : self.fv[k] - self.fe[k]}))
            ys.append(self(x, **{k : self.fv[k] + self.fe[k]}))
        return (np.min(ys, axis=0), np.max(ys, axis=0))
    def plot_fitdata(self, ax=None, x=None):
        '''Plot data and the fitline on ax (or new figure) with x (or self.xp) for fitline'''
        if x is None: x = self.xp
        if ax is None: fig, ax = plt.subplots()
        ax.errorbar(self.x, self.y, self.y_err, fmt='r.-')
        ax.plot(x, self(x), 'k')
        return ax
    def plot_residuals(self, ax=None):
        '''Plot residual with vertical lines and zero line on ax (or new figure)'''
        if ax is None: fig, ax = plt.subplots()
        ax.axhline(0, c='k', alpha=0.5)
        ax.vlines(self.x, self.x*0, self.y-self())
        ax.plot(self.x, self.y-self(), 'r.')
        return ax
    def plot_residuals_hist(self, ax=None, orientation='vertical'):
        '''Plot histogram of the residul on ax (or new figure) with orientation either vertical or horizontal'''
        if ax is None: fig, ax = plt.subplots()
        ax.hist(self.y-self(), orientation=orientation)
        return ax
    def plot_fiterrors(self, ax=None, x=None, using=[]):
        '''Plot a band of y representing fit errors : on ax (or a new figure) with x (or self.ax) and with using list (or all) of fit variables'''
        if x is None: x = self.xp
        ax = self.plot_fitdata(ax)
        ax.fill_between(x, *self.yband(x=x, using=using), color='g', alpha=0.25)
        return ax
    def plot(self, ax=None, x=None, fiterrors=True, using=[], divider=0.25):
        '''Plot data with fitline and '''
        if ax is None: ax = plt.subplots(figsize=[5,5])[1]
        if x is None: x = self.xp
        (ax1, ax2) = divide_axes(ax, divider=divider, direction='vertical', shared=True)
        if fiterrors: self.plot_fiterrors(ax=ax1, x=x, using=using)
        else: self.plot_fitdata(ax=ax1, x=x)
        self.plot_residuals(ax2)
        return (ax1, ax2)

'''
Area of Partial ellipse
=======================
'''
@np.vectorize
def area_partial_ellipse(A, a, b=None):
    '''
    Comput area of ellipse/cicrle chopped off on either sides
    =========================================================
    inputs : A, a, b=None
    Ellipse with size a, b in the horizontal and verical dimension
    b=None referes to circle and b = a
    Chop off the ellipse from -A to A in the horizontal direction
    '''
    # for an ellipse with horizontal length a and vertical length b. Area from -A to A in horizontal
    if b is None: b = a
    if A >= a: return np.pi*a*b
    return 2*A*b*np.sqrt(1-(A/a)**2) + 2*a*b*np.arcsin(A/a)

################################################################################
################################################################################
# C8 : GUI Related
################################################################################

class qTextEditDictIO:
    '''
    Data format for settings
    qSett (dict)
        keys: name of the setting
        vals: list of QTextEdit
    sett (dict)
        keys: name of the setting
        vals: float value for the setting

    '''
    def __init__(self, qSett=None, func=None, filepath=None):
        self.qSett = qSett
        self.func = func
        self.filepath = filepath
        self.sett = self.setupValuesFloatFromFile(func,filepath)
        self.connectEditSignals()


    def getValuesFloat(self):
        for key in self.qSett.keys():
            self.sett[key] = float(self.qSett[key].text())

    def setupValuesFloatFromFile(self, func=None, filepath=None):
        self.sett = dictio(filepath=filepath)        # Get default values from file
        # Change the text to default value
        for key in self.qSett.keys():
            self.qSett[key].setText(str(self.sett.get(key,'nan')))

    def connectEditSignals(self):
        for key in self.qSett.keys():
            self.qSett[key].editingFinished.connect(self.getValuesFloat)


################################################################################
################################################################################
# C9 : Legacy Functions
################################################################################

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
        self.cst = kwargs.get('cst', cst())
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
            self.var['rawod'] = get_od(self.wa, self.woa, rawod=True)
        return self.var['rawod']

    @property
    def fixod(self):
        if 'fixod' not in self.var.keys():
            rawod = get_od(self.wa, self.woa, width=self.var.get('trouble_pts_width', 5), rawod=False)
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
        cropi = get_cropi(self.od, **kwargs)
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
        fy = c.fit(ThomasFermi_harmonic, guess, plot=False)[0]
        # along x
        c = Curve(y=np.nansum(self.od, axis=0))
        max_y = np.max(c.y[c.y.shape[0] // 4:3 * c.y.shape[0] // 4])
        ind = np.argwhere(c.y == max_y)[0][0]
        guess = [c.x[ind], c.x.shape[0] / 10, c.y[ind], c.y[ind] / 10, c.y[ind] / 100]
        fx = c.fit(ThomasFermi_harmonic, guess, plot=False)[0]
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
        return get_usable_pixels(self.wa, self.woa, threshold=threshold)

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
        tck = scipy.interpolate.spintp.splrep(self.ycen_, self.r_, s=100)
        self.rad = scipy.interpolate.spintp.splev(self.yall, tck, der=0)
        self.rad[self.yall < self.ycen_[0]] = scipy.interpolate.spintp.splev(self.ycen_[0], tck, der=0)
        self.rad[self.yall > self.ycen_[-1]] = scipy.interpolate.spintp.splev(self.ycen_[-1], tck, der=0)
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

class ODFix2D:
    # Removing OD gradients in cropped image
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
        self.od_ = fix_od(self.od_, width=5)

    def fit(self):
        guess = [0, 0, 0]
        self.fitres, self.fiterr = scipy.optimize.curve_fit(self.fitfun_2DPoly, self.xy, self.z, p0=guess)
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

class OD2Density:
    # Convert OD to Density
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
        fitres, fiterr = c.fit(ThomasFermi_harmonic, guess, plot=False)
        y = c.y - (ThomasFermi_harmonic(c.x, fitres[0], fitres[1], 0, fitres[3], fitres[4]))
        if self.center is None: self.center = fitres[0]
        self.nz = Curve(x=(c.x - self.center) * self.pixel, y=y, xscale=1e-6)
        guess = [self.pixel, fitres[1]*self.pixel, fitres[2], fitres[3], fitres[4]/self.pixel]
        self.nz.fit(ThomasFermi_harmonic, guess, plot=False)

    def infoplot(self):
        fig, ax1 = plt.subplots(figsize=(4, 3))
        ax1.scatter(self.nz.x * 1e6, self.nz.y,color='red',alpha=0.5,marker='.',s=7)
        ax1.plot(self.nz.xyfitplot[0]*1e6,self.nz.xyfitplot[1],'k-')

class hybridEoS_avg:
    '''
    kwargs -- required
        cst = tp.cst(sigmaf=0.5, Nsat=200, pixel=0.68e-6, trapw=2*np.pi*23.9)
        cropi or [center=(x,y), width=400, height=400]
    kwargs -- optional
        bgwidth = 20
        odlim or [odlimA = (0,2), odlimB = (0,2)]
        bgplot = False
        Xyuse or [XyuseA = None, XyuseB = None]  this will be selected by fitting TF profile and yset
        Xmethod or [XmethodA = 'linear', XmethodB = 'linear']
        Xplot or [XplotA = False, XplotB = False]
        Xyset or [XysetA = (10, 0.75), XysetB = (10, 0.75)]  fit circles to 75% of the center and 10 pixels average
        Xguess or [XguessA = None, XguessB = None]

        usecenter = 'A', 'B', or None, default is None
        Ncor or [NcorA = 1, NcorB = 1]
        nmax or [nmaxA = np.inf, nmaxB = np.inf]


    '''
    def __init__(self, names, **kwargs):
        self.var = kwargs
        self.processNames(names)
        self.computeOD()

    @property
    def cst(self):
        return self.var['cst']

    @property
    def cropi(self):
        return self.var['cropi']

    @property
    def odfixs(self):
        if 'odfixA' not in self.var.keys():
            self.fixOds()
        return (self.var['odfixA'], self.var['odfixB'])

    @property
    def xsecs(self):
        if 'xsecA' not in self.var.keys():
            self.getXsecs()
        return (self.var['xsecA'], self.var['xsecB'])

    @property
    def odA(self):
        return self.xsecs[0].od

    @property
    def odB(self):
        return self.xsecs[1].od

    @property
    def nzA(self):
        if 'nzA' not in self.var.keys():
            self.OD2Densities()
        return self.var['nzA']

    @property
    def nzB(self):
        if 'nzB' not in self.var.keys():
            self.OD2Densities()
        return self.var['nzB']

    @property
    def nuA(self):
        u = 0.5 * self.cst.mass * self.cst.trapw ** 2 * self.nzA.x ** 2
        return Curve(x=u, y=self.nzA.y, xscale=self.cst.h * 1e3, yscale=1).sortbyx()

    @property
    def nuB(self):
        u = 0.5 * self.cst.mass * self.cst.trapw ** 2 * self.nzB.x ** 2
        return Curve(x=u, y=self.nzB.y, xscale=self.cst.h * 1e3, yscale=1).sortbyx()

    @property
    def EFuA(self):
        u = 0.5 * self.cst.mass * self.cst.trapw ** 2 * self.nzA.x ** 2
        return Curve(x=u, y=self.cst.n2EF(self.nzA.y, True),
                             xscale=self.cst.h*1e3, yscale=self.cst.h*1e3).sortbyx()

    @property
    def EFuB(self):
        u = 0.5 * self.cst.mass * self.cst.trapw ** 2 * self.nzB.x ** 2
        return Curve(x=u, y=self.cst.n2EF(self.nzB.y, True),
                             xscale=self.cst.h*1e3, yscale=self.cst.h*1e3).sortbyx()

    def processNames(self, names):
        # The format of input 'names'
        if type(names) is str:
            namesA = [names]
            namesB = [names[0:-1] + 'B']
        elif type(names) is list:
            namesA = names[0::2]
            namesB = names[1::2]
        elif type(names).__module__ == 'pandas.core.series':
            names_ = names.tolist()
            namesA = names_[0::2]
            namesB = names_[1::2]
        else:
            raise ValueError("Invalid input for names")
        self.var['namesA'] = namesA
        self.var['namesB'] = namesB

    def computeOD(self):
        # Get the first OD
        namesA = self.var['namesA']
        namesB = self.var['namesB']
        imA = AbsImage(name=namesA[0], cst=self.cst)
        imB = AbsImage(name=namesB[0], cst=self.cst)
        odA = imA.od
        odB = imB.od
        # Average rest of the images
        for i, nA in enumerate(namesA[1:]):
            odA += AbsImage(name=namesA[i], cst=self.cst).od
            odB += AbsImage(name=namesB[i], cst=self.cst).od
        odA /= len(namesA)
        odB /= len(namesB)
        # Get cropi
        if 'cropi' not in self.var.keys():
            self.var['cropi'] = imA.cropi(**self.var)
        # Store
        self.var['odA'] = odA
        self.var['odB'] = odB

    def fixOds(self):
        if 'odlim' in self.var.keys():
            self.var['odlimA'], self.var['odlimB'] = self.var['odlim'], self.var['odlim']
        odfixA = ODFix2D(od=self.var['odA'],
                                 cropi=self.cropi,
                                 width=self.var.get('bgwidth', 20),
                                 odlim=self.var.get('odlimA', (0, 2)),
                                 plot=self.var.get('bgplot', False))
        odfixB = ODFix2D(od=self.var['odB'],
                                 cropi=self.cropi,
                                 width=self.var.get('bgwidth', 20),
                                 odlim=self.var.get('odlimB', (0, 2)),
                                 plot=self.var.get('bgplot', False))
        self.var['odfixA'] = odfixA
        self.var['odfixB'] = odfixB

    def getXsecs(self):
        # process inputs
        if 'Xyuse' in self.var.keys():
            self.var['XuseA'], self.var['XuseB'] = self.var['Xyuse'], self.var['Xyuse']
        if 'Xmethod' in self.var.keys():
            self.var['XmethodA'], self.var['XmethodB'] = self.var['Xmethod'], self.var['Xmethod']
        if 'Xplot' in self.var.keys():
            self.var['XplotA'], self.var['XplotB'] = self.var['Xplot'], self.var['Xplot']
        if 'odlim' in self.var.keys():
            self.var['odlimA'], self.var['odlimB'] = self.var['odlim'], self.var['odlim']
        if 'Xyset' in self.var.keys():
            self.var['XysetA'], self.var['XysetB'] = self.var['Xyset'], self.var['Xyset']
        if 'Xguess' in self.var.keys():
            self.var['XguessA'], self.var['XguessB'] = self.var['Xguess'], self.var['Xguess']

        # calculate cross sections
        self.var['xsecA'] = XSectionTop(od=self.odfixs[0].od,
                                                yuse=self.var.get('XyuseA', None),
                                                method=self.var.get('XmethodA', 'linear'),
                                                plot=self.var.get('XplotA', False),
                                                odlim=self.var.get('odlimA', (0, 2)),
                                                yset=self.var.get('XysetA', (10, 0.75)),
                                                guess=self.var.get('XguessA', None))
        self.var['xsecB'] = XSectionTop(od=self.odfixs[1].od,
                                                yuse=self.var.get('XyuseB', None),
                                                method=self.var.get('XmethodB', 'linear'),
                                                plot=self.var.get('XplotB', False),
                                                odlim=self.var.get('odlimB', (0, 2)),
                                                yset=self.var.get('XysetB', (10, 0.75)),
                                                guess=self.var.get('XguessB', None))

    def OD2Densities(self):
        # Process inputs
        if 'Ncor' in self.var.keys():
            NcorA = self.var['Ncor']
            NcorB = NcorA
        else:
            NcorA = self.var.get('NcorA', 1)
            NcorB = self.var.get('NcorB', 1)
        if 'nmax' in self.var.keys():
            self.var['nmaxA'], self.var['nmaxB'] = self.var['nmax'], self.var['nmax']

        if self.var.get('usecenter', None) == 'A':
            self.var['od2denA'] = OD2Density(od=self.odA, xsec=self.xsecs[0], pixel=self.cst.pixel,
                                                     sigma=self.cst.sigma, nmax=self.var.get('nmaxA', np.inf),
                                                     Ncor=NcorA, plot=False, center=None)
            self.var['od2denB'] = OD2Density(od=self.odB, xsec=self.xsecs[0], pixel=self.cst.pixel,
                                                     sigma=self.cst.sigma, nmax=self.var.get('nmaxB', np.inf),
                                                     Ncor=NcorB, plot=False, center=self.var['od2denA'].center)
        elif self.var.get('usecenter', None) == 'B':
            self.var['od2denB'] = OD2Density(od=self.odB, xsec=self.xsecs[0], pixel=self.cst.pixel,
                                                     sigma=self.cst.sigma, nmax=self.var.get('nmaxB', np.inf),
                                                     Ncor=NcorB, plot=False, center=None)
            self.var['od2denA'] = OD2Density(od=self.odA, xsec=self.xsecs[0], pixel=self.cst.pixel,
                                                     sigma=self.cst.sigma, nmax=self.var.get('nmaxA', np.inf),
                                                     Ncor=NcorA, plot=False, center=self.var['od2denB'].center)
        else:
            self.var['od2denA'] = OD2Density(od=self.odA, xsec=self.xsecs[0], pixel=self.cst.pixel,
                                                     sigma=self.cst.sigma, nmax=self.var.get('nmaxA', np.inf),
                                                     Ncor=NcorA, plot=False, center=None)
            self.var['od2denB'] = OD2Density(od=self.odB, xsec=self.xsecs[0], pixel=self.cst.pixel,
                                                     sigma=self.cst.sigma, nmax=self.var.get('nmaxB', np.inf),
                                                     Ncor=NcorB, plot=False, center=None)

        self.var['nzA'] = self.var['od2denA'].nz
        self.var['nzB'] = self.var['od2denB'].nz

    def infoplot(self, odlimA=None, odlimB=None):
        if 'odlim' in self.var.keys():
            self.var['odlimA'], self.var['odlimB'] = self.var['odlim'], self.var['odlim']
        if odlimA is None: odlimA = self.var.get('odlimA', (0, 2))
        if odlimB is None: odlimB = self.var.get('odlimB', (0, 2))

        fig = plt.figure(figsize=(15, 4))
        ax = plt.subplot2grid((4, 3), (0, 0), rowspan=4)
        axt = plt.subplot2grid((4, 3), (0, 1), rowspan=4)
        ax1 = plt.subplot2grid((4, 3), (0, 2))
        ax2 = plt.subplot2grid((4, 3), (1, 2))
        ax3 = plt.subplot2grid((4, 3), (2, 2), rowspan=2)

        ax.scatter(self.nzA.x * 1e6, self.nzA.y, color='blue', alpha=0.4, marker='.', s=7)
        ax.plot(self.nzA.xyfitplot[0] * 1e6, self.nzA.xyfitplot[1], 'k-')
        ax.scatter(self.nzB.x * 1e6, self.nzB.y, color='red', alpha=0.4, marker='.', s=7)
        ax.plot(self.nzB.xyfitplot[0] * 1e6, self.nzB.xyfitplot[1], 'k-')
        ax.plot([0, 0], [min(self.nzA.miny, self.nzB.miny), 1.1 * max(self.nzA.maxy, self.nzB.maxy)], 'k--')
        ax.set(xlabel='z [$\mu$m]', ylabel='n [m$^{-3}$]')

        axt.plot(*self.EFuA.plotdata, 'b-', alpha=0.6)
        axt.plot(*self.EFuB.plotdata, 'r-', alpha=0.6)
        axt.plot([0, 1.1 * np.max(self.EFuA.plotdata[0])], [0, 0], 'k--')
        axt.set( xlabel='U [kHz]', ylabel='$E_F$ [kHz]')

        ax1.scatter(self.xsecs[0].ycen_, self.xsecs[0].xl_, s=1, color='white')
        ax1.scatter(self.xsecs[0].ycen_, self.xsecs[0].xr_, s=1, color='white')
        ax1.scatter(self.xsecs[0].ycen_, self.xsecs[0].xc_, s=1, color='k')
        ax1.imshow(self.xsecs[0].od.T, clim=odlimA, aspect='auto', cmap='viridis', origin='lower')
        ax1.set(xlim=[self.xsecs[0].yall[0], self.xsecs[0].yall[-1]], title='OD and radius')
        ax1.set_axis_off()

        ax2.scatter(self.xsecs[1].ycen_, self.xsecs[1].xl_, s=1, color='white')
        ax2.scatter(self.xsecs[1].ycen_, self.xsecs[1].xr_, s=1, color='white')
        ax2.scatter(self.xsecs[1].ycen_, self.xsecs[1].xc_, s=1, color='k')
        ax2.imshow(self.xsecs[1].od.T, clim=odlimB, aspect='auto', cmap='viridis', origin='lower')
        ax2.set(xlim=[self.xsecs[1].yall[0], self.xsecs[1].yall[-1]])
        ax2.set_axis_off()

        ax3.plot(self.xsecs[0].yall, self.xsecs[0].rad, 'k-')
        ax3.scatter(self.xsecs[0].ycen_, self.xsecs[0].r_, color='blue')
        ax3.set(xlim=[self.xsecs[0].yall[0], self.xsecs[0].yall[-1]])
        ax3.plot(self.xsecs[1].yall, self.xsecs[1].rad, 'k-')
        ax3.scatter(self.xsecs[1].ycen_, self.xsecs[1].r_, color='red')
        ax3.set(xlim=[self.xsecs[1].yall[0], self.xsecs[1].yall[-1]])
        ax3.set(xlabel='z [pixel]', ylabel='radius [pixels]')

    def getThermoPlot(self,**sett):
        # Bin EFu
        EFuA = self.EFuA.binbyx(bins = sett.get('binsA', (20,10)),
                                edges = sett.get('edgesA', self.EFuA.maxx/3))
        EFuB = self.EFuB.binbyx(bins = sett.get('binsB', (20,10)),
                                edges = sett.get('edgesB', self.EFuB.maxx/3))
        # Get kappa
        kuA = EFuA.diff(method='poly',order=sett.get('dorder',1), points=sett.get('dpoints',1))
        kuA = Curve(kuA.x, -kuA.y, xscale=self.cst.h*1e3, yscale=1)
        kuB = EFuB.diff(method='poly', order=sett.get('dorder', 1), points=sett.get('dpoints', 1))
        kuB = Curve(kuB.x, -kuB.y, xscale=self.cst.h*1e3, yscale=1)

        # store
        self.thermo = dict(EFuA=EFuA, EFuB=EFuB, kuA = kuA, kuB = kuB)

        # Plot
        fig,ax = plt.subplots(ncols=2,figsize=(10,4))
        ax[0].plot(*self.EFuA.plotdata,'b-',alpha=0.3)
        ax[0].plot(*EFuA.plotdata,'b.')
        ax[0].plot(*self.EFuB.plotdata, 'r-', alpha=0.3)
        ax[0].plot(*EFuB.plotdata, 'r.')
        ax[0].plot([0, 1.1*EFuA.maxx/EFuA.xscale],[0,0],'k--')
        ax[0].set(xlabel='U [kHz]', ylabel='$E_F$ [kHz]')

        ax[1].plot(*kuA.plotdata, 'b-')
        ax[1].plot(*kuA.plotdata, 'b.')
        ax[1].plot(*kuB.plotdata, 'r-')
        ax[1].plot(*kuB.plotdata, 'r.')
        ax[1].set(xlabel='U [kHz]', ylabel='$\kappa / \kappa_0$')


################################################################################
################################################################################
# C0 : Housekeeping
################################################################################
'''
Download LookupTable Data, and precompile it for interp_od
'''
p_ = getpath('Projects','Data','LookupTable','Lookup_Table_Fast_PreCompData_V2.p')
if not os.path.isfile(p_):
    print("Downloading Lookup Database -- Might take some time!")
    url = "https://www.dropbox.com/s/4hklnvawtshjay9/Lookup_Table_Fast_PreCompData_V2.p?dl=1"
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    # Create folder
    os.makedirs(os.path.split(p_)[0], exist_ok=True)
    with open(p_, "wb") as f :
        f.write(data)
precompiled_data_Lookup_Table = pickle.load( open( p_, "rb" ) )


'''
Download Mark EoS Density Generator
'''
p_ = getpath('Projects','Data','EoS','Mark_Density_EoS_Extended_Data4Python.p')
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

'''
Download Li6 scattering lengths
'''
p_ = getpath('Projects','Data','EoS','Li6_scattering_length_jochim_julienne_2013_Data4Python.p')
if not os.path.isfile(p_):
    print("Downloading scattering lengths")
    url = "https://www.dropbox.com/s/i01berhdhavh882/scatteringLi6.p?dl=1"
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    # Create folder
    os.makedirs(os.path.split(p_)[0], exist_ok=True)
    with open(p_, "wb") as f :
        f.write(data)
scattering_length_data = pickle.load( open( p_, "rb" ) )
a_12_interp = scipy.interpolate.interp1d(scattering_length_data['B'], scattering_length_data['a12'])
a_23_interp = scipy.interpolate.interp1d(scattering_length_data['B'], scattering_length_data['a23'])
a_13_interp = scipy.interpolate.interp1d(scattering_length_data['B'], scattering_length_data['a13'])

'''
Initialize cst_## objects
'''
cst_LiD2 = cst(atom='LiD2')
cst_LiD1 = cst(atom='LiD1')
cst_NaD2 = cst(atom='NaD2')
cst_NaD1 = cst(atom='NaD1')
cst_ = cst_LiD2
# Special constants needed throughout the different codes
cst_.ldB_prefactor = ((cst_.twopi * cst_.hbar**2)/(cst_.mass))**(1/2)
cst_.xi = 0.37
cst_.virial_coef = [1, 3*2**(1/2)/8, -0.29095295, 0.065]
cst_.ideal_gas_density_prefactor = 1/(6*np.pi**2) * (2*cst_.mass/cst_.hbar**2)**(3/2)
twopi = 2 * np.pi
pi = np.pi
kHz = 1e3 * cst_.h


'''
Warnings for Users
'''
# print('''Internal Structure of therpy has changed.
# Now all functions (except for helper functions and imageio functions) are stored at one place therpy.funcs and available on root as tp.####.
# If there are errors in your program, simply replace tp.module_name.func_name to tp.func_name.
# If some function is not present there, it must be a helper function and can be accessed via tp.funcs._____
# If you would like to go back to old therpy, use pip install therpy==0.2.5 ''')
