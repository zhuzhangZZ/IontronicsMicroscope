# -*- coding: utf-8 -*-
"""
LandingCheck.py 
Look at if and where particles are landing.

Settings are a bit finicky. 
Read up on the trackpy package online to understand how to set these settings.

@author: Kevin Namink <k.w.namink@uu.nl>

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import trackpy

# To import our library with functions you might need to put the functionsDFSM.py file in the same folder as this file
import DFSM as dfsm


# Settings
# Configure settings, make folders, open data files and declare some functions:

info = """
Put info here, like from the readme file associated with the measurement being
analysed, so it is always also available.
"""

# Configure:
folder = "/media/kevin/My Passport/2019-04-11-measurements/"
filename = "2nd_PolyLLystine-preAu"
movienumber = 1
extension = ".hdf5"
needsbinning = True  # Bins 2by2 to increase further analysis speed.

#%%
# Set figure as a popout to enable it to play like a movie:
# NOTE: don't close the figure before it is done, it might break the program.
%matplotlib auto  
# This line changes a setting in spyder (can also be done using the preferences menu)
#%matplotlib inline  # Alternate option

#%% 
# Import data and quickly show some properties of the file

# Import data
data = dfsm.ImportHDF5data(folder+filename+extension)
signaldata = np.load(folder+filename+"_m"+str(movienumber)+".npy")

# Quickly plot a small part of the data
xrange, yrange, Nmax = data[movienumber,1].shape
middleline = np.mean(data[movienumber,1][int(xrange/2-10):int(xrange/2+10),int(yrange/2),:], axis=0)
startframe = 0
endframe = np.argwhere(middleline==0)[0,0]
plt.plot(middleline[startframe:endframe])
plt.title("Average of the center line")
plt.xlabel("Frame number")
plt.ylabel("Intensity (arb.u.)")


#%%
# Edit startframe and endframe if you want to:
# Possibly skip the first frames because there is an LED flashing.
# If the endframe is very far away from the startframe calculations take a very long time.
startframe = 25
endframe = endframe-25

plt.plot(middleline[startframe:endframe])
plt.title("Average of the center line")
plt.xlabel("Frame number")
plt.ylabel("Intensity (arb.u.)")


#%%
# Do background correction
# This block manually averages 6 frames before the current frame
# and substracts it from the current frame. This results in a particular 
# shape of the movie when a particle lands, which can be detected:
# Frame 1: 100% intensity
# Frame 2: 80% intensity
# Frame 3: 60% intensity
# Frame 4: 40% intensity
# Frame 5: 20% intensity
# Frame 6+: 0% intensity

# Help variables:
startframe2 = 6  # +6 for accounting for the moving average
endframe2 = endframe-startframe
movie = np.array(data[movienumber,1][:,:,startframe:endframe])
# Bin figure:
if needsbinning == True:
    movie2 = np.zeros((int(movie.shape[0]/2),int(movie.shape[1]/2),movie.shape[2]))
    for i in range(movie2.shape[0]):
        for j in range(movie2.shape[1]):
            movie2[i,j,:] = movie[2*i, 2*j, :] + movie[2*i+1, 2*j, :] + movie[2*i, 2*j+1, :] + movie[2*i+1, 2*j+1, :]
else:
    movie2 = movie
# Optionally use weights on the five bgcorrection frames:
# Make background substracted movie where background is a moving average of 5 previous frames.
bgcorrected = movie2[:,:,startframe2:endframe2] - np.mean([movie2[:,:,startframe2-5:endframe2-5],
                                                           movie2[:,:,startframe2-4:endframe2-4],
                                                           movie2[:,:,startframe2-3:endframe2-3],
                                                           movie2[:,:,startframe2-2:endframe2-2],
                                                           movie2[:,:,startframe2-1:endframe2-1]] , axis = 0)
# Normalize, with the first frame of the movie as the normalization:
bgcorrected = bgcorrected/movie2[:,:,0,None]

# Change index order from x,y,t to t,y,x to be able to use trackpy
b = np.reshape(np.ravel(bgcorrected, order='C'), bgcorrected.shape[::-1], order='F')

#%%
# Show resulting backgroundcorrected movie, a flash should be a landing particle
plt.figure(figsize=(10,10))
im=plt.imshow(bgcorrected[:,:,0])
plt.colorbar()
for i in np.arange(1,bgcorrected.shape[2]):
    im.set_data(bgcorrected[:,:,i])
    plt.pause(1./33)
plt.show()


#%%
# Run trackpy finder:
# Google trackpy for more information on trackpy

f = trackpy.batch(b, (15,31), minmass=2) 

t = trackpy.link_df(f[f['mass']==f['mass']], 3, memory=1)
t2 = trackpy.filter_stubs(t, 4)

#%%
# Show figure of landing canditates:

plt.figure(figsize=(10,10))
plt.title("All found particle track locations")
trackpy.annotate(t2, (movie2[:,:,-1]-movie2[:,:,0]).T);
plt.show()

#%%
# Filter landing candidates as described before (with the linear decay).
# Note: this is not very efficiently or clearly coded.

interestingpathslist = []
for i in t2.groupby('particle'):  # Filter paths that are too short:
    x = next(iter(i[1]['frame'].to_dict().values()))
    if ( i[1]["mass"][x] > 5 ):
        interestingpathslist.append(i[1])
        
masslist = []
locationslist = []
decaylist = []
timelist = []
for i in interestingpathslist:  # Take interesting data from still accepted paths:
    x = next(iter(i['frame'].to_dict().values()))
    masslist.append(i['mass'][x])
    locationslist.append([i['x'][x],i['y'][x]])
    decayrate = i['mass']/i['mass'][x]
    decaylist.append(decayrate)
    timelist.append(i['frame'][x])

acceptedparticles=[]
for i in range(len(masslist)):  # Filter for expected decay shape:
    ddict = decaylist[i]
    m = masslist[i]
    l = decaylist[i]
    d = []
    for j in ddict.to_dict().values():
        d.append(j)
    # Interesting particle landings decay as follows:
    if (d[1]-0.8)**2>0.01:
        continue
    if (d[2]-0.6)**2>0.01:
        continue
    if (d[3]-0.4)**2>0.01:
        continue
    acceptedparticles.append(i)

"""So 'acceptedparticles' is now a list of indices for the 
'masslist', 'locationslist', 'decaylist' and 'timelist'
wherin the information of these landings is mostly saved.
"""

#%%
# Show accepted particles in a movie:

plt.figure(figsize=(10,10))
im=plt.imshow(bgcorrected[:,:,-1].T)
for i in acceptedparticles:
    plt.scatter(locationslist[i][0], locationslist[i][1], s=100, facecolors='none', edgecolors='r')

for i in np.arange(1,bgcorrected.shape[2]):
    im.set_data(bgcorrected[:,:,i].T)
    plt.pause(1./10)
plt.show()


#%%
# Trying to plot this nicely:

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import (AnchoredSizeBar)
from matplotlib import font_manager as fm
SIZE = 14
pixelsize = 6.5/157  # 6.5 is the pixel size on camera and 157 magnification of the microscope.


plt.figure(figsize=(10,7))
fig = plt.imshow((data[movienumber,1][:,:,endframe]-data[movienumber,1][:,:,startframe]).T, clim=(0,700), cmap='Greys')

cb1 = plt.colorbar(fig)
cb1.set_label('background corrected intensity', size=SIZE)

fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

bar = AnchoredSizeBar(fig.axes.transData, (10/pixelsize), '10 $\mu$m', 4, borderpad=0.5, size_vertical=10, frameon=False)
fig.axes.add_artist(bar)

for i in acceptedparticles:
    plt.scatter(locationslist[i][0]*2, locationslist[i][1]*2, s=100, facecolors='none', linewidth=2, edgecolors='r')

plt.show()



#%%
# Another possibly nice plot:

fps = 33.
SIZE = 15

plt.figure(figsize=(12,6))

plt.subplot(121)
for i in acceptedparticles:
    p = np.array(decaylist[i])
    plt.plot(np.arange(len(p))/fps, p)

plt.xlabel("Time /s", size = SIZE)
plt.ylabel("relative intensity", size = SIZE)
plt.tick_params(labelsize=SIZE-1) 

plt.text(-0.065, 1, "(a)", size=SIZE)
plt.text( 0.298, 1, "(b)", size=SIZE)


plt.subplot(122)
m = []
for i in acceptedparticles:
    m.append(masslist[i])
    if m[-1]>40:
        print(i)
plt.hist(m, bins=12, edgecolor='black', linewidth=1)

plt.xlabel("total intensity", size = SIZE)
plt.ylabel("counts", size = SIZE)
plt.tick_params(labelsize=SIZE-1) 

plt.show()



