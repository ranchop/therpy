# Please put the version of therpy that you like to use a (for Mac and Windows)
# ‘Documents/My Programs/Python Library’
# Note that ‘Documents/My Programs’ is used for various function, 
# including imageio to store downloaded images. It is highly recommended that 
# you use this folder scheme for the uniformity and cross compability of our 
# programs. 

### Method to import therpy
import os.path, sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'Documents', 'My Programs', 'Python Library'))
import therpy


### Beginning of jupyter notebook
# Housekeeping
import os.path, sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'Documents', 'My Programs', 'Python Library'))
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import therpy as tp
# Matplotlib backend interactive
import matplotlib
matplotlib.use('nbagg')
# Matplotlib backend static
# %matplotlib inline
