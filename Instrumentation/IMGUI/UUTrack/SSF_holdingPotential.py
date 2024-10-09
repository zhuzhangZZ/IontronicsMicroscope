# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 21:29:37 2023

@author: iSCAT
"""

import hardpotato as hp
import softpotato as sp
import matplotlib.pyplot as plt
import os 
# Select the potentiostat model to use:
#model = 'chi760e'
model = 'chi760e'
#model = 'emstatpico'

# Path to the chi software, including extension .exe. Negletected by emstatpico
path = 'D:/Equipment/CHI/chi760e/chi760e.exe'

# Folder where to save the data, it needs to be created previously
folder = 'C:/Data/Zhu/2023-08-03_Pd/CHI_data'

if not os.path.exists(folder):
    os.makedirs(folder)
    print('the folder of %s has created'%folder)
else:
    print('the folder of %s already exists '%folder)


Eholding = -1.1
holdingTime = 20
Eend = 0
ScanRates = 0.02
# Initialization:
hp.potentiostat.Setup(model, path, folder)

# Experimental parameters:
ExtTri = False
# tech=ssf  #select Sweep-Step Functions
Eini = 0 # initial potential in V
dE = 0.001 # sweep sample interval in V
dt = 0.01 # step sample interval in s
qt = 10 # quiescent time before run in s
sens =1e-5 #sensitivity in A/V
Eswi1 = Eholding # initial potential in V for Sequence 1: Sweep
Eswf1 = Eholding # final potential in V for Sequence 1: Sweep
sr1 = ScanRates  #1e-4   -   10 scan rate in V/s for Sequence 1: Sweep
Estep1 = Eholding #  -10   -   +10 step potential in V for Sequence 2: Step
tstep1 = holdingTime # 0   -   10000 step time in s for Sequence 2: Step
Eswi2 = Eholding # -10  -  +10 initial potential in V for Sequence 3: Sweep
Eswf2 = Eend #10   -   +10 final potential in V for Sequence 3: Sweep
sr2 = ScanRates #   -   10 scan rate in V/s for Sequence 3: Sweep
fileName = 'Video1_m14' # base file name for data file
header = 'SSF'   # header for data file

# Initialize experiment:
SSF = hp.potentiostat.SSF(Eini, dE, dt, sens, Eswi1, Eswf1, sr1, Estep1, tstep1, Eswi2, Eswf2, sr2, \
                         fileName, header, ExtTri, qt=qt)
# Run experiment:
SSF.run()

# Load recently acquired data
data = hp.load_data.SSF(fileName +'.txt', folder, model)
i = -data.i
E = data.E

# Plot CV with softpotato
sp.plotting.plot(E, i, fig=1, show=1)
plt.plot(i)
