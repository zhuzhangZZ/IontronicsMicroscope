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
folder = "/media/kevin/My Passport/2019-03-21-measurements/"
filename = "02_Triangle_1p0Vpp_NOGNP"
movienumber = 1

Tframe = 1/200.  # In seconds the time per frame 
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

movieframe = np.argmax(middleline[:30])
signalframe = np.argmax(signaldata[:,2])
moviesignaloffset = movieframe - signalframe

if False:
    for i in range(10):
        plt.figure(figsize=(8,6))
        plt.imshow(movie[:,:,i+1].T, clim = (0, 5000))
        plt.colorbar()
        plt.show()
        print("Frame %d with signal %lf." %(i, signaldata[i-moviesignaloffset, 2]))
if True:    
    plt.figure(figsize=(8,6))
    plt.plot(meanintime[0:20]/np.max(meanintime[0:20]), label="Movie mean")
    plt.plot(signaldata[0:20,2]/np.max(signaldata[0:20,2]), 'r:', label="Signal to LED")
    plt.title("Pre-synchronisation")
    plt.legend()
    if save_automatically:
        plt.savefig(dirName+"/overview/Pre-Synch_"+filename)
    plt.show()
if True:
    plt.figure(figsize=(8,6))
    plt.plot(signaldata[:,0])
    plt.title("Voltage over sample per frame")
    if save_automatically:
        plt.savefig(dirName+"/overview/signalVoltage_"+filename)
    plt.show()
    plt.figure(figsize=(8,6))
    plt.plot(signaldata[:,1])
    plt.title("'Current' trough sample per frame")
    if save_automatically:
        plt.savefig(dirName+"/overview/signalCurrent_"+filename)
    plt.show()
    plt.figure(figsize=(8,6))
    plt.plot(signaldata[:,2])
    plt.title("Signal to LED")
    if save_automatically:
        plt.savefig(dirName+"/overview/signaltoLED_"+filename)
    plt.show()

print("Found offset (movie - signal):", moviesignaloffset)
if moviesignaloffset>0:
    meanintime = meanintime[moviesignaloffset:]
    signaldata = signaldata[:,:]
else:
    meanintime = meanintime[:]
    signaldata = signaldata[-moviesignaloffset:,:]

if False:
    for i in range(10):
        plt.figure(figsize=(18,4))
        plt.imshow(movie[:,:,i+1].T-movie[:,:,i].T, clim = (-5000,5000))
        plt.colorbar()
        plt.show()
        print(i)
if True:
    plt.figure(figsize=(8,6))
    plt.plot(meanintime[0:20]/np.max(meanintime[0:20]), label="Movie mean")
    plt.plot(signaldata[0:20,2]/np.max(signaldata[0:20,2]), 'r:', label="Signal to LED")
    plt.title("Post synchronisation (offset=%d)"%moviesignaloffset)
    plt.legend()
    if save_automatically:
        plt.savefig(dirName+"/overview/Post-Synch_"+filename)
    plt.show()














