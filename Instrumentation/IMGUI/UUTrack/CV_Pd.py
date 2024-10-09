# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import hardpotato as hp
import softpotato as sp
import os

# Select the potentiostat model to use:
#model = 'chi760e'
model = 'chi760e'
#model = 'emstatpico'

# Path to the chi software, including extension .exe. Negletected by emstatpico
path = 'D:/Equipment/CHI/chi760e/chi760e.exe'
# Folder where to save the data, it needs to be created previously
folder = 'C:/Data/Zhu/2023-08-11_Pd'
if not os.path.exists(folder):
    os.makedirs(folder)
    print('the folder of %s has created'%folder)
else:
    print('the folder of %s already exists '%folder)
    

Estart = -0.4
Eend = -0.9
ScanRate = 2

# Initialization:f
hp.potentiostat.Setup(model, path, folder)

# Experimental parameters:
ExtTri = False  
Eini = 0   # V, initial potential
Ev1 = -0.2 # V, first vertex potential
Ev2 = -1.2   # V, second vertex potential
Efin = -0.2  # V, finad potential
sr = ScanRate        # V/s, scan rate
dE = 0.001      # V, potential increment
nSweeps = 4   # number of sweeps
sens = 1e-4  # A/V, current sensitivity
# E2 = 0.5        # V, potential of the second working electrode
# sens2 = 1e-9    # A/V, current sensitivity of the second working electrode
fileName = 'Video9_m0' # base file name for data file
header = 'CV'   # header zfor data file

# Initialize experiment:
cv = hp.potentiostat.CV(Eini, Ev1,Ev2, Efin, sr, dE, nSweeps, sens, fileName, header, ExtTri)
# Run experiment:
cv.run()

# Load recently acquired data
data = hp.load_data.CV(fileName +'.txt', folder, model)
i = -data.i
E = data.E

# Plot CV with softpotato
sp.plotting.plot(E, i, fig=1, show=1)
#%%
import matplotlib.pyplot as plt
plt.plot(i)
plt.ylim(None,None)