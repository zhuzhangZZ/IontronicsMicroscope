# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 13:18:34 2023

@author: Zhu Zhang <z.zhang@uu.nl>

"""
#%% CV
import hardpotato as hp
import softpotato as sp

# Select the potentiostat model to use:
#model = 'chi760e'
model = 'chi760e'
#model = 'emstatpico'

# Path to the chi software, including extension .exe. Negletected by emstatpico
path = 'D:/Equipment/CHI/chi760e/chi760e.exe'

# Folder where to save the data, it needs to be created previously
folder = 'C:/Users/iSCAT/OneDrive/Desktop/CHI_data'

# Initialization:
hp.potentiostat.Setup(model, path, folder)

# Experimental parameters:
Eini = -0.5     # V, initial potential
Ev1 = 0.5       # V, first vertex potential
Ev2 = -0.5      # V, second vertex potential
Efin = -0.5     # V, final potential
sr = 0.1        # V/s, scan rate
dE = 0.001      # V, potential increment
nSweeps = 2     # number of sweeps
sens = 1e-6     # A/V, current sensitivity
E2 = 0.5        # V, potential of the second working electrode
sens2 = 1e-9    # A/V, current sensitivity of the second working electrode
fileName = 'CV_1' # base file name for data file
header = 'CV'   # header for data file

# Initialize experiment:
cv = hp.potentiostat.CV(Eini, Ev1,Ev2, Efin, sr, dE, nSweeps, sens, fileName, header)
# Run experiment:
cv.run()

# Load recently acquired data
data = hp.load_data.CV(fileName +'.txt', folder, model)
i = data.i
E = data.E

# Plot CV with softpotato
sp.plotting.plot(E, i, fig=1, show=1)


#%% CV_ScanRate

import hardpotato as hp
import numpy as np
import matplotlib.pyplot as plt
import softpotato as sp

# Select the potentiostat model to use:
model = 'chi760e'
#model = 'chi760e'
# Path to the chi software, including extension .exe
path = 'D:/Equipment/CHI/chi760e/chi760e.exe'

# Folder where to save the data, it needs to be created previously
folder = 'C:/Users/iSCAT/OneDrive/Desktop/CHI_data'
# Initialization:
hp.potentiostat.Setup(model, path, folder)


# Experimental parameters:
Eini = -0.5     # V, initial potential
Ev1 = 0.5       # V, first vertex potential
Ev2 = -0.5      # V, second vertex potential
Efin = -0.5     # V, final potential
dE = 0.001      # V, potential increment
nSweeps = 2     # number of sweeps
sens = 1e-6     # A/V, current sensitivity
E2 = 0.5        # V, potential of the second working electrode
sens2 = 1e-9    # A/V, current sensitivity of the second working electrode
header = 'CV'   # header for data file

sr = np.array([0.2, 0.5, 1])          # V/s, scan rates
nsr = sr.size

for x in range(nsr):
    # initialize experiment:
    fileName = 'CV_' + str(int(sr[x]*1000)) + 'mVs'# base file name for data file
    print(fileName)
    cv = hp.potentiostat.CV(Eini, Ev1,Ev2, Efin, sr[x], dE, nSweeps, sens, fileName, header)
    # Include second working electrode in bipotentiostat mode.
    # Comment or delete the next line to remove bipot mode.
    #cv.bipot(E2,sens2)
    # Run experiment:
    cv.run()
    
#%% RandlesSevcik.py

import hardpotato as hp
import numpy as np
import matplotlib.pyplot as plt
import softpotato as sp
from scipy.optimize import curve_fit

##### Setup
# Select the potentiostat model to use:
# emstatpico, chi1205b, chi760e
#model = 'chi760e'
model = 'chi760e'
#model = 'chi760e'
# Path to the chi software, including extension .exe
path = 'D:/Equipment/CHI/chi760e/chi760e.exe'

# Folder where to save the data, it needs to be created previously
folder = 'C:/Users/iSCAT/OneDrive/Desktop/CHI_data'
# Initialization:
hp.potentiostat.Setup(model, path, folder)


##### Experimental parameters:
Eini = -0.3     # V, initial potential
Ev1 = 0.5       # V, first vertex potential
Ev2 = -0.3      # V, second vertex potential
Efin = -0.3     # V, final potential
dE = 0.001      # V, potential increment
nSweeps = 2     # number of sweeps
sens = 1e-6     # A/V, current sensitivity
header = 'CV'   # header for data file

##### Experiment:
sr = np.array([0.2, 0.5, 0.1, 0.2])          # V/s, scan rate
nsr = sr.size
i = []
for x in range(nsr):
    # initialize experiment:
    fileName = 'CV_' + str(int(sr[x]*1000)) + 'mVs'# base file name for data file
    cv = hp.potentiostat.CV(Eini, Ev1,Ev2, Efin, sr[x], dE, nSweeps, sens, fileName, header)
    # Run experiment:
    cv.run()
    # load data to do the data analysis later
    data = hp.load_data.CV(fileName + '.txt', folder, model)
    i.append(data.i)
i = np.array(i)
i = i[:,:,0].T
E = data.E


##### Data analysis
# Estimation of D with Randles-Sevcik
n = 1       # number of electrons
A = 0.071   # cm2, geometrical area
C = 1e-6    # mol/cm3, bulk concentration

# Showcases how powerful softpotato can be for fitting:
def DiffCoef(sr, D):
    macro = sp.Macro(n, A, C, D)
    rs = macro.RandlesSevcik(sr)
    return rs
    
iPk_an = np.max(i, axis=0)
iPk_ca = np.min(i, axis=0)
iPk = np.array([iPk_an, iPk_ca]).T
popt, pcov = curve_fit(DiffCoef, sr, iPk_an)
D = popt[0]

# Estimation of E0 from all CVs:
EPk_an = E[np.argmax(i, axis=0)]
EPk_ca = E[np.argmin(i, axis=0)]
E0 = np.mean((EPk_an+EPk_ca)/2)

#### Simulation with softpotato
iSim = []
for x in range(nsr):
    wf = sp.technique.Sweep(Eini,Ev1, sr[x])
    sim = sp.simulate.E(wf, n, A, E0, 0, C, D, D)
    sim.run()
    iSim.append(sim.i)
iSim = np.array(iSim).T
print(iSim.shape)
ESim = sim.E

##### Printing results
print('\n\n----------Results----------')
print('D = {:.2f}x10^-6 cm2/s'.format(D*1e6))
print('E0 = {:.2f} mV'.format(E0*1e3))

##### Plotting
srsqrt = np.sqrt(sr)
sp.plotting.plot(E, i*1e6, ylab='$i$ / $\mu$A', fig=1, show=0)
sp.plotting.plot(srsqrt, iPk*1e6, mark='o-', xlab=r'$\nu^{1/2}$ / V$^{1/2}$ s$^{-1/2}$', 
                 ylab='$i$ / $\mu$A', fig=2, show=0)

plt.figure(3)
plt.plot(E, i*1e6)
plt.plot(wf.E, iSim*1e6, 'k--')
plt.title('Experiment (-) vs Simulation (--)')
sp.plotting.format(xlab='$E$ / V', ylab='$i$ / $\mu$A', legend=[0], show=1)