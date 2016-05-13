import numpy as np

# Library of functions



# Thomas - Fermi profile 
# 0 temperature ideal fermi gas and unitary gas
def Thomas_Fermi_harmonic(r, r0, rmax, amp):
	return np.nan_to_num(np.real(  amp * (1-((r-r0)/rmax)**2)**(3/2)  ))

# Gaussian 1D
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
