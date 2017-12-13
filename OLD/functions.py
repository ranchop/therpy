import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Importing from therpy is its main else using .
if __name__ == '__main__':
	import os.path
	import sys
	path2therpy = os.path.join(os.path.expanduser('~'), 'Documents', 'My Programs', 'Python Library')
	sys.path.append(path2therpy)
	from therpy import constants
else:
	from . import constants

# Library of functions



# Thomas - Fermi profile 
# 0 temperature ideal fermi gas and unitary gas
def Thomas_Fermi_harmonic(r, r0, rmax, amp, offset):
	return (np.nan_to_num(np.real(  amp * (1-((r-r0)/rmax)**2)**(3/2)  )) + offset)

def ThomasFermiCentering(*args,**kwargs):
    # Input parser
    guess = kwargs.get('guess',None)
    plot = kwargs.get('plot',False)
    output = kwargs.get('output',False)
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
    center_x = x[np.equal(y,max_y)][0]
    ym = y / max_y
    xm = x.copy()
    # Guess
    guess = [center_x, x.size/2, 1, 0] # r0, rmax, amp, offset
    # Fiting
    res,_ = curve_fit(Thomas_Fermi_harmonic,xm,ym,guess)
    # New x and y
    xn = xm - res[0]
    yn = (ym - res[3]) * max_y
    # Plot and output
    if plot:
        if max_y > 1e5:
            print('Thomas Fermi fit gave center {:.1f}, radius {:.1f}, amplitude {:.1e}, and offset {:.1e}'.format(res[0],res[1],res[2]*max_y,res[3]*max_y))
        else:
            print('Thomas Fermi fit gave center {:.1f}, radius {:.1f}, amplitude {:.1f}, and offset {:.1f}'.format(res[0],res[1],res[2]*max_y,res[3]*max_y))
        fig,axes = plt.subplots(figsize=(8,3),ncols=2)
        axes[0].plot(x,y,x,Thomas_Fermi_harmonic(x,*res)*max_y)
        axes[0].set(title='Original Data and Fit')
        axes[1].plot(xn,yn,xn,Thomas_Fermi_harmonic(x,res[0],res[1],res[2]*max_y,0))
        axes1_2 = axes[1].twinx()
        axes1_2.plot(xn,yn-Thomas_Fermi_harmonic(x,res[0],res[1],res[2]*max_y,0),'r')
        axes[1].set(title='Centered and offseted with error')
    # Outp
    if output:
        outp = {'info':'ThomasFermiCentering Function did its magic'}
        outp['fitres'] = res
        outp['center'], outp['radius'] = res[0], res[1]
        outp['amp'], outp['offset'] = res[2]*max_y, res[3]*max_y
        if plot: outp['fig'] = fig
        return (xn,yn,outp)
    else:
        return(xn,yn)

def FermiDirac(beta,mu,k):
	return 1 / (1 + np.exp( beta*( (((1.0545718001391127e-34)**2 * k**2) / (2 * 9.988346 * 10**-27)) - mu) ))

def betamu2n(beta,mu):
	# Find k_max when f(k_max) < 1e-3
	k_max, fk_max = 1, FermiDirac(beta, mu, 1)
	while fk_max > 1e-3: k_max, fk_max = k_max*10, FermiDirac(beta, mu, k_max * 10)
	dk = k_max / 1e5
	# Integrate
	k = np.arange(0,k_max,dk)
	fk = FermiDirac(beta, mu, k)
	return 1/(2*np.pi**2) * np.sum( k*k*fk ) * dk

def FermiDiracFit(k,fk,**kwargs):
	# Input parser
	plot = kwargs.get('plot',False)
	cst = kwargs.get('cst',constants.cst())
	kF_atoms = kwargs.get('kF',k[k.size//2])
	EF_atoms = cst.kF2EF(kF_atoms)
	TF_atoms = EF_atoms / cst.kB
	# Prepare variables
	usepts = np.isfinite(fk)
	km, fkm = k[usepts], fk[usepts]
	# Rescale k
	km = km / kF_atoms
	# Fit function
	def FDFit(k,T,mu,amp):
		return amp / (1 + np.exp(1/T *(k**2-mu)))
	# Guess
	guess = [0.2,0.9,1]
	# Fit
	res,_ = curve_fit(FDFit,km,fkm,guess)
	# Useful quantities
	outp = {'info':'Howdy! from FermiDiracFit function'}
	outp['T'] = res[0] * EF_atoms / cst.kB
	outp['beta'] = 1 / (outp['T'] * cst.kB)
	outp['mu'] = res[1] * EF_atoms
	outp['n'] = betamu2n(outp['beta'],outp['mu'])
	outp['kF'] = cst.n2kF(outp['n'])
	outp['EF'] = cst.n2EF(outp['n'])
	outp['EFHz'] = outp['EF'] / cst.h
	outp['TF'] = outp['EF'] / cst.kB
	outp['TTF'] = outp['T'] / outp['TF']
	outp['muEF'] = outp['mu'] / outp['EF']
	x = np.linspace(0,2,1000)
	y = FermiDirac(outp['beta'],outp['mu'],x*outp['kF']) * res[2]
	outp['plotdata'] = (x, y)
	outp['disp'] = 'Fermi-Dirac fit results: T = {:.2f} TF; kF = {:2f} M m^-1; EF = {:.2f} kHz; mu = {:.2f} EF'.format(
					outp['TTF'], outp['kF']*1e-6, outp['EFHz']*1e-3, outp['muEF'])
	# plot
	if plot:
		print(outp['disp'])
		fig,axes = plt.subplots(figsize = (4,4))
		axes.plot(k/outp['kF'],fk,'r.')
		axes.plot(x,y)
		outp['fig'] = fig
	return outp



def rabi_resonance(**kwargs):
	'''
	Two-photon rabi resonance function

	Inputs:
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
	if 'tau' in keys: tau = kwargs['tau']
	elif 't' in keys: tau = kwargs['t']
	else: print("ERROR: pulse duration must be provided using keys 't' or 'tau'")
	# omega rabi
	if 'wR' in keys: omega_Rabi = kwargs['wR']
	elif 'fR' in keys: omega_Rabi = kwargs['fR'] * cst.twopi
	elif 'wr' in keys: omega_Rabi = kwargs['wr']
	elif 'fr' in keys: omega_Rabi = kwargs['fr'] * cst.twopi
	else: print("ERROR: rabi frequency must be provided using keys 'fr' 'fR' 'wr' 'wR'")
	# delta
	if 'delta' in keys: delta = kwargs['delta'] * cst.twopi
	elif 'd' in keys: delta = kwargs['d'] * cst.twopi
	elif 'df' in keys: delta = kwargs['df'] * cst.twopi
	else: # delta is not provided
		# resonance frequency
		if 'w0' in keys: omega_0 = kwargs['w0']
		elif 'f0' in keys: omega_0 = kwargs['f0'] * cst.twopi
		else: print("ERROR: resonance frequency must be provided using keys 'f0' 'w0' since delta wasn't provided")
		# applied frequency
		if 'w' in keys: delta = kwargs['w'] - omega_0
		elif 'f' in keys: delta = kwargs['f'] * cst.twopi - omega_0
		else: print("ERROR: rabi frequency must be provided using keys 'fr' 'fR' 'wr' 'wR'")
	# Compute and return
	omega_eff_2 = delta**2 + omega_Rabi**2
	return omega_Rabi**2 / omega_eff_2 * np.sin(np.sqrt(omega_eff_2)*tau/2)**2


def main():
	print('Hi from main')

if __name__ == '__main__':
	main()