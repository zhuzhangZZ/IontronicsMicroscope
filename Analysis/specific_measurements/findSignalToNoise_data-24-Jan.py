# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 09:50:16 2019

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import functionsDFSM as dfsm

# Import data:
data = dfsm.ImportHDF5data("/home/kevin/Documents/PDSM_data/2019-01-24_different_salts/Video_NaCl_1Hz_55fps_particles.hdf5")
data.setkey(0)
print("Wherein there is : ", data.getkeys())
data.resetkeys()

datab = dfsm.ImportHDF5data("/home/kevin/Documents/PDSM_data/2019-01-24_different_salts/Snap_NaCl_1Hz_55fps_refrence.hdf5")
datab.setkey(0)
print("Wherein there is : ", datab.getkeys())
datab.resetkeys()

"""
Using this file:
    Copy it to a logical place and rename the copy something logical
    You also need functionsDFSM.py (so add where that file is to the python root for example)
    Go block by block to find the settings you need
    Get what you want
    Profit

Syntax in this file:
    Start block of code
    Comments that say what the block does
    Variables that can/should be changed by the user
    ...
    Code that should work 
    ...
    
    Next block
"""



#%%
# Look at "framenumber" frame of "movienumber" movie to test if settings setup correctly:
framenumber = 5
movienumber = 1
refpicture_n = 7  # Up to 8 

picture = data[movienumber,1][:,:,framenumber]
plt.subplot(121)
plt.imshow(picture)
plt.title("Intensity with particles")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()

refpicture = datab[refpicture_n, 0][:]
refpicture = data[0,1][:,:,framenumber]
plt.subplot(122)
plt.imshow(refpicture)
plt.title("Intensity without particles")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()

plt.show()


plt.imshow((picture - refpicture))  #, clim=(-5000, 5000))
plt.title("Difference between with and without particles")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()

plt.show()



#%%
# Plot the difference between the two images and select areas to look at:
yofinterest, xofinterest = 583, 218

plt.imshow((picture - refpicture))  #, clim=(-5000, 5000))
plt.title("Difference between with and without particles")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()

plt.scatter(xofinterest, yofinterest, s=2, c='red', marker='o')


plt.show()





















