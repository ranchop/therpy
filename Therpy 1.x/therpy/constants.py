# Import constants from scipy.constants
import scipy.constants as spconstants
import numpy as np

## Temporary constants
_TRAP_OMEGA = 2 * np.pi * 23.9
_TRAP_VOLUME = np.pi * (70e-6) ** 2 * 80e-6


class cst:
    '''
	Constants to use for computations. 
	Atomic specifications provided by kwargs.
	atom = 'Li', 'LiD2', 'LiD1', 'Na', 'NaD1', 'NaD2'  [Na and NaD2 are same, default: LiD2]
	sigmaf = 0.5 for top and 1 for side
	Nsat = 120 (from calibration for particular imaging time)
	Ncor = 1 (atom number fudge)
	ODf = 1 (OD prefactor)
	pixel = 1.39e-6 (the size of pixel onto atoms)
	trapw = 2*pi*23.9 (Trapping frequency)
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

    # fill in atom specific params
    def LiD2(self):
        self.f = 446.799677e12
        self.tau = 27.102e-9
        self.mass = 9.988346 * 10 ** -27
        self.atomtype = 'Lithium 6, D2 Line'

    def LiD1(self):
        self.f = 446.789634e12
        self.tau = 27.102e-9
        self.mass = 9.988346 * 10 ** -27
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
        self.h = spconstants.h
        self.hbar = spconstants.hbar
        self.pi = spconstants.pi
        self.c = spconstants.c
        self.mu_0 = spconstants.mu_0
        self.epsilon_0 = spconstants.epsilon_0
        self.G = spconstants.G
        self.g = spconstants.g
        self.e = spconstants.e
        self.R = spconstants.R
        self.alpha = spconstants.alpha
        self.N_A = spconstants.N_A
        self.kB = spconstants.k
        self.Rydberg = spconstants.Rydberg
        self.m_e = spconstants.m_e
        self.m_p = spconstants.m_p
        self.m_n = spconstants.m_n
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
        self.pixel = self.var.get('pixel', 1.39e-6)
        self.Ncor = self.var.get('Ncor', 1)
        self.ODf = self.var.get('ODf', 1)
        self.trapw = self.var.get('trapw', 2 * np.pi * 23.9)
        self.radius = self.var.get('radius', 70e-6)
        self.width = self.var.get('width', 80e-6)
        self.volume = self.var.get('volume', np.pi * self.radius ** 2 * self.width)

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

    def kT2lambdaDB(self, kT, inHz = False):
        if inHz: kT *= self.h
        return self.h / (self.twopi * self.mass * kT)**(1/2)

    ## Very specific conversions
    # Fermi-Surface conversions
    def z2k(self, z):
        return z * (self.mass * self.trapw / self.hbar)

    def N2f(self, N):
        return N * self.h / (self.volume * self.mass * self.trapw)

    def df1D2f(self, df1D):
        return df1D * (-4 * self.pi)


