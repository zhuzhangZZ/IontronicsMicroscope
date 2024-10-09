# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 09:50:16 2019

@author: kevin
Zhu Zhang
"""

import numpy as np
import matplotlib.pyplot as plt

# To import our library with functions you might need to put 
# the DFSM.py file in the same folder as this file
import DFSM as dfsm

# Import data:
folder = "C:\\Data\\Zhu\\2023-05-30_Sample\\"
filename = "LED4"
movienumber = 0
extension = ".hdf5"
framerate = 50.  # In frames per second
Tframe = 1/framerate

#%%


# Camera data:
data = dfsm.ImportHDF5data(folder+filename+extension)
xrange, yrange, startframe, endframe, quicklook = \
    dfsm.movie_properties(data, movienumber)
mean_img = np.mean(data[movienumber,1][:,:,
                   startframe:endframe], axis=0)
average = np.mean(mean_img, axis=0)

#%%

import numpy as np
import matplotlib.pyplot as plt

# To import our library with functions you might need to put 
# the DFSM.py file in the same folder as this file
import DFSM as dfsm

# Import data:
folder = "C:\\Data\\Zhu\\2023-05-30_Sample\\"
filename = "LED3"
movienumber = 0
extension = ".hdf5"
framerate = 50.  # In frames per second
Tframe = 1/framerate

#%%


# Camera data:
data = dfsm.ImportHDF5data(folder+filename+extension)
xrange, yrange, startframe, endframe, quicklook = \
    dfsm.movie_properties(data, movienumber)
mean_img = np.mean(data[movienumber,1][:,:,
                   startframe:endframe], axis=0)
average2 = np.mean(mean_img, axis=0)





#%%
# Look at "framenumber" frame of "movienumber" movie and show all data tracks normalized:

fig, ax1 = plt.subplots(figsize=(10,5))
color = 'xkcd:red'
ax1.set_xlabel('$t$ (s)', size=18)
ax1.set_ylabel('$potential$ (V)', size=18, color=color) 
ax1.plot(average[50:2000], color=color)
ax1.tick_params(axis='y', labelsize=14, labelcolor=color)

ax2 = ax1.twinx()
color = 'xkcd:green'
ax2.set_ylabel('$current$ (mA)', size=18, color=color)
ax2.plot(average2[50:2000], color=color)
ax2.tick_params(axis='y', labelsize=14, labelcolor=color)
    



#%%







