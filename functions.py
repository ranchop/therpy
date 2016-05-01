import numpy as np

# Library of functions



# Thomas - Fermi profile 
# 0 temperature ideal fermi gas and unitary gas
def Thomas_Fermi_harmonic(r, r0, rmax, amp):
	return np.nan_to_num(np.real(  amp * (1-((r-r0)/rmax)**2)**(3/2)  ))

# Gaussian 1D
