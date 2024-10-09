# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 09:50:16 2019

@author: Kevin Namink <k.w.namink@uu.nl>

Feedback and Comments by Sanli Faez
"""


import numpy as np
import matplotlib.pyplot as plt
import os

# To import our library with functions you might need to put the functionsDFSM.py file in the same folder as this file
import functionsDFSM as dfsm


def createFolders():
    # Create folders
    global dirName
    if savedirName == 0:
        dirName = folder+filename+"_m"+str(movienumber)+"/"
        if not os.path.exists(dirName):
            os.makedirs(dirName)
    else:
        dirName = savedirName+"/"
        if not os.path.exists(dirName):
            os.makedirs(dirName)
    for f in ["/overview"]:
        if not os.path.exists(dirName+f):
            os.makedirs(dirName+f)


def customSave(figurename, plotted, raw = True):
    # Needs to be called when the figure is in "memory" needs a logical name of the figure and the plotted np data.
    # Use to save the figure currently in use and the raw np data used to make it, puts it in the correct folder.
    if save_automatically: 
        if raw:
            previewfile = dirName+"/"+figurename+"_"+filename+'_rawImage.npy'
        np.save(previewfile, plotted)
        plt.savefig(dirName+"/"+figurename+"_"+filename)

def getvariablesandmiddleline():
    # Calculate time averaged intensity and variance/intensity of movie:
    global FPsignal, xrange, yrange, startframe, endframe, nf, middleline
    
    # Autosetting some variables:
    FPsignal = Tsignal/Tframe  
    xrange, yrange, Nmax = data[movienumber-1,1].shape
    # Find and plot average of middle line, a 'fast' way to see if the data is nice:
    middleline = np.mean(data[movienumber-1,1][:,int(yrange/2),:], axis=0)
    nf = np.argwhere(middleline==0)[0,0]
    plt.plot(middleline[:nf])
    plt.title("Average of the center line")
    plt.xlabel("Frame number")
    plt.ylabel("Intensity (arb.u.)")
    if save_automatically: 
        plt.savefig(dirName+"overview/Average-Center-Line_"+filename)
    plt.show()
    # Autoset some more variables and report on found values:
    startframe = 0
    endframe = nf
    Nperiods = (endframe-startframe)/FPsignal  # This variable is used for knowing how many peaks to find for the self correlation
    print("startframe: %d \nendframe: %d \nxrange: 0 to %d \nyrange: 0 to %d \nNperiods: %lf (%d frames per signal in %d frames)" %(startframe, endframe, xrange, yrange, Nperiods, FPsignal, endframe-startframe))


def getmeanIandvarI():
    # Calculate mean intensity and variance/intensity of movie:
    global safelength, mean_img, var_img, var_img_MOD, movie
    
    # Throw away part of the data if the data is too long to process. 
    safelength = 5000  # I cannot put this higher or I get memory issues..
    
    # Assign 'movie' array which holds al the data we want in a nice to work way:
    movieTimeLength = endframe-startframe
    safeendframe, safestartframe = endframe, startframe
    if movieTimeLength > safelength:
        safeendframe = endframe - int((movieTimeLength-safelength)/2)
        safestartframe = startframe + int((movieTimeLength-safelength)/2)
    movie = np.array(data[movienumber-1,1][:,:,safestartframe:safeendframe])
    # Calculate mean while substracting dark field:
    mean_img = np.mean(movie, axis=2)
    dark = np.full(mean_img.shape, 96)  # Measured dark field is 96
    dark[dark>mean_img] = mean_img[dark>mean_img] # Set dark field to be lower than usual where the mean image has lower valued pixels, to fix dividing by zero
    mean_img = mean_img - dark + 1
    del dark
    # Calculate variance:
    var_img = np.var(movie, axis=2)/mean_img
    print("Max and min of Var I/I:", np.max(var_img), np.min(var_img))
    # Modify variance for some calculations:
    var_img_MOD = var_img.copy()
    var_img_MOD[var_img<1] = 1  # To get rid of saturated pixels that will show sub-shot noise variance
    var_img_MOD[var_img>100] = 1  # To get rid of exceptionally irregular points
    print("Filtered: " + str((np.sum(var_img>100)+np.sum(var_img<1))/var_img.size) + " %")
    # Show mean:
    plt.figure(figsize=(18,4)) 
    dfsm.plotlogMean(mean_img)
    if save_automatically:
        customSave("overview/Time-Averaged-Picture", mean_img)
    plt.show()

#%%
# Settings
# Configure settings, make folders, open data files and declare some functions:

# Configure:
folder = "/media/kevin/My Passport/2019-04-11-measurements/"
filename = "2nd_PolyLLystine-preAu"
movienumber = 1

Tframe = 1/33.  # In seconds the time per frame 
hertz = 1
Tsignal = 1./hertz  # In seconds
Asignal = 1.  # In volts
savedirName = folder+filename+"_m"+str(movienumber-1)+"/"
save_automatically = True

extension = ".hdf5"

#%% 
# Do things to the whole chosen movie:

createFolders()

# Import data
data = dfsm.ImportHDF5data(folder+filename+extension)
data.setkey(0)
print("Wherein there is : ", data.getkeys())
data.resetkeys()
signaldata = np.load(folder+filename+"_m"+str(movienumber-1)+".npy")

getvariablesandmiddleline()

getmeanIandvarI()

#%%
# Temporary:

if False:
    for i in range(10):
        plt.imshow(np.log10(movie[:,:,i+150]).T, clim=(1,4))
        plt.colorbar()
        plt.show()
        print(i)


if True:
    interval = int(len(movie[1,1,:])/10)
    print(interval)
    for i in range(10):
        plt.imshow((movie[:,:,i*interval]-movie[:,:,0]).T, clim=(-100,100))
        plt.colorbar()
        plt.show()
        print(i)

if False:
    for i in range(50):
        ii = i+150
        plt.imshow((movie[:,:,ii]-movie[:,:,150]).T, clim=(-10,100))
        plt.colorbar()
        plt.show()
        print(i)

#%%

plt.plot(middleline)
plt.show()



#%%
withoutGNP = np.mean(movie[:,:,150:155], axis=2)
withGNP = np.mean(movie[:,:,350:355], axis=2)

plt.imshow(withoutGNP.T, clim=(96,500))
plt.title("Without GNP (averaged)")
plt.colorbar()
customSave("overview/withoutGNP", withoutGNP)
plt.show()
plt.imshow(withGNP.T, clim=(96,500))
plt.title("With GNP (averaged)")
plt.colorbar()
customSave("overview/withGNP", withGNP)
plt.show()
difference = withGNP-withoutGNP
plt.imshow((withGNP-withoutGNP).T, clim=(-100,100))
plt.title("Difference")
plt.colorbar()
customSave("overview/differenceGNP", withGNP-withoutGNP)
plt.show()

#loaded = np.load(folder+"NoGNP_m0/overview/differenceGNP_NoGNP_rawImage.npy")
#plt.hist(np.reshape(loaded, -1), bins = 200, range=(-200,200) )
#plt.savefig(dirName+"/"+"differenceGNP_dist"+"_"+filename)
#plt.show()


#%%

differenceed = np.log10(withGNP.T)#(difference.T)

import trackpy

f = trackpy.locate(differenceed, (11,31), minmass=15)

plt.figure(figsize=(10,10)) 
trackpy.annotate(f, differenceed);

fig, ax = plt.subplots()
ax.hist(f['mass'], bins=30)

ax.set(xlabel='mass', ylabel='count');










