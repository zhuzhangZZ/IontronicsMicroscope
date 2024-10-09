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


def getSignalperiod():
    global FPsignal, offsetsignal, amplitudesignal
    global signaldata, signal_potential, signal_current, signal_LED
    signaldata = np.load(folder+filename+"_m"+str(movienumber-1)+".npy")
    signal_potential = signaldata[:,0]
    signal_current = signaldata[:,1]
    signal_LED = signaldata[:,2]
    ff = np.fft.rfft(signal_potential)
    fr = len(signal_potential)/np.max([np.argmax(np.abs(ff)),1])
    start = np.argmax(signal_potential[:int(fr)])
    argmaxlist = [start]
    while(argmaxlist[-1]+fr*1.25 < len(signal_potential)):
        argmaxlist.append(int(argmaxlist[-1]+fr*0.75)+np.argmax(signal_potential[int(argmaxlist[-1]+fr*0.75):int(argmaxlist[-1]+fr*1.25)]))
    FPsignal = np.mean(np.diff(argmaxlist))
    offsetsignal = np.mean(argmaxlist%FPsignal)
    amplitudesignal = np.mean(signal_potential[argmaxlist])

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
    print("startframe: %d \nendframe: %d \nxrange: 0 to %d \nyrange: 0 to %d \nNperiods: %lf (%lf frames per signal in %d frames)" %(startframe, endframe, xrange, yrange, Nperiods, FPsignal, endframe-startframe))


def getmeanIandvarI():
    # Calculate mean intensity and variance/intensity of movie:
    global safelength, mean_img, var_img, var_img_MOD, movie, safestartframe, safeendframe
    
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

def spotfinder_initial():
    # Guess some spots to look at
    global interestingspots
    
    binnedforfindingstuffvar = dfsm.doBin(dfsm.doBin(var_img_MOD))
    binnedforfindingstuffint = dfsm.doBin(dfsm.doBin(mean_img))
    hvarx, hvary = np.unravel_index(np.argmax(binnedforfindingstuffvar*binnedforfindingstuffint), binnedforfindingstuffvar.shape)
    lvarx, lvary = np.unravel_index(np.argmax(binnedforfindingstuffint**(0.5)/binnedforfindingstuffvar), binnedforfindingstuffvar.shape)
    dvarx, dvary = np.unravel_index(np.argmin(binnedforfindingstuffvar*binnedforfindingstuffint), binnedforfindingstuffvar.shape)
    kvarx, kvary = np.unravel_index(np.product([ord(l) for l in 'Kevin'])%mean_img.size, mean_img.shape)
    svarx, svary = np.unravel_index(np.product([ord(l) for l in 'Sanli'])%mean_img.size, mean_img.shape)
    
    interestingspots = np.array([('ITO-specle', 1, 1),
                                 ('High-variance_high-intensity', hvarx*4, hvary*4),
                                 ('Low-variance_high-intensity', lvarx*4, lvary*4),
                                 ('Dark-area_no-var-no-int', dvarx*4, dvary*4),
                                 ('Random_spot', kvarx, kvary),
                                 ('Random_spot2', svarx, svary)], dtype=[('def', np.unicode_, 100), ('x', int), ('y', int)])
    plt.figure(figsize=(18,5))
    dfsm.plotlogMean(mean_img)
    for i in range(6):
        plt.scatter(interestingspots[i][1], interestingspots[i][2], s=5, label=interestingspots[i][0])
    plt.legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=1.)
    plt.show()


def spotfinder(lookat, x = 0, y = 0):
    # Find spots of interest
    global interestingspots
    if x==0 and y==0:
        plt.figure(figsize = (16,9))
        dfsm.plotlogMeanlogVarwithROI(mean_img, var_img_MOD, interestingspots[lookat])
        print("%s remains set to x=%d y=%d" %(interestingspots[lookat][0],interestingspots[lookat][1],interestingspots[lookat][2]))
    else:
        interestingspots[lookat][1] = x
        interestingspots[lookat][2] = y
        plt.figure(figsize = (16,9))
        dfsm.plotlogMeanlogVarwithROI(mean_img, var_img_MOD, interestingspots[lookat])
        print("%s set to x=%d y=%d" %(interestingspots[lookat][0],x,y))

def spotfinder_final():
    # Guess some spots to look at
    global interestingspots
    
    plt.figure(figsize=(18,5))
    dfsm.plotlogMean(mean_img)
    for i in range(6):
        plt.scatter(interestingspots[i][1], interestingspots[i][2], s=5, label=interestingspots[i][0])
    plt.legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=1.)
    plt.show()
    
def harmonic_components():
    global dcbase, maxFFTc, safeendframe, safestartframe, particlexsize, particleysize
    
    for lookat in range(len(interestingspots)):
        
        # First make folders for the spots just identified:
        for f in interestingspots[:]['def']:
            if not os.path.exists(dirName+f):
                os.makedirs(dirName+f)
                
        # Do fast fourier transform on area around spot
        x, y = interestingspots[lookat]['x'], interestingspots[lookat]['y']
        xmin, xmax, ymin, ymax = dfsm.getROIaroundXYinFigure(x, y, movie[:,:,0])
        FFTspecROI = np.fft.rfft(movie[xmin:xmax,ymin:ymax], axis=2)/(safeendframe - safestartframe)
        # Find maximum valued frequency:
        meanfftspec = np.mean(np.mean(FFTspecROI[x-xmin-particlexsize:x-xmin+particlexsize, y-ymin-particleysize:y-ymin+particleysize, :], axis=0), axis=0)
        mfreq = dcbase + np.argmax(abs(meanfftspec[dcbase:maxFFTc]))
        shfreq = mfreq + 1 + np.argmax(abs(meanfftspec[mfreq+1:maxFFTc]))
        thirdfreq = shfreq + 1 + np.argmax(abs(meanfftspec[shfreq+1:maxFFTc]))
        # Plot fourier components:
        plt.figure(figsize=(10,6))
        plt.plot(np.arange(dcbase, maxFFTc), np.log10(abs(meanfftspec[dcbase:maxFFTc]))) # Give correct "x"axis
        plt.title("Fourier components of specle at x=%d y=%d" %(x, y))
        plt.xlabel("Fourier component")
        plt.ylabel("Amplitude? (log arb.u.)")
        if save_automatically:
            figurename = interestingspots[lookat]['def']+"/FFTspectrum"
            customSave(figurename, FFTspecROI)
        plt.show()
        print("Fourier components resp. first second third:", mfreq, shfreq, thirdfreq)
        print("Movie was %d long so %lf frames per period of this component" %(len(movie[0,0,:]), len(movie[0,0,:])/mfreq))
        print("(Skipped fourier components dcbase=%d and expected from applied signal fc=%lf)" %(dcbase, len(movie[0,0,:])/FPsignal))
        
        
        # Plot harmonic and stuff:
        
        plotfreq = mfreq  # Option to look at different frequency
        ffpwidth = 3  #estimated half width of the Fourier peak in the power spectrum
        
        # Find and plot first harmonic and stuff:
        first_harmonic = np.sum(np.abs(FFTspecROI[:, :, plotfreq-ffpwidth:plotfreq+ffpwidth+1]), axis=2)
        mean_img_SOI = mean_img[xmin:xmax,ymin:ymax]
        plt.figure(figsize=(18,12))
        dfsm.plotFHandmean(first_harmonic, mean_img_SOI, plotfreq, (xmin,xmax,ymax,ymin))
        if save_automatically:
            figurename = interestingspots[lookat]['def']+"/First_harmonic"
            customSave(figurename, first_harmonic)
        plt.show()
        
        
        # Looking at the phase:
        phase = np.angle(FFTspecROI[:,:, plotfreq])/np.pi
        dfsm.getPhaseIntensityColormapped(phase.T, mean_img_SOI.T, (xmin,xmax,ymax,ymin))
        if save_automatically:
            figurename = interestingspots[lookat]['def']+"/phase"
            customSave(figurename, phase)
        plt.show()
        print(interestingspots[lookat]['def'], "finished.")
        
        
def spotandsignal(lookat, lowlim=0, highlim=0):
    global interestingspots, mof, sof, meanoverparticle, msmax
    x, y = interestingspots[lookat][1], interestingspots[lookat][2]
    movieframe = np.argmax(middleline[:30])
    signalframe = np.argmax(signaldata[:,2])
    moviesignaloffset = movieframe - signalframe
    print("Found offset (movie - signal):", moviesignaloffset)
    if abs(moviesignaloffset)>20:
        print("Found offset too large to be accurate, using old data if available:")
        print("movie-signal =>", mof-sof )
        print(np.max(signaldata[:,2]), " is the maximum LED signal registered")
    elif moviesignaloffset>0:
        mof = moviesignaloffset
        sof = lowlim
    else:
        mof = lowlim
        sof = - moviesignaloffset
    msmax = min(len(signal_potential[:]),len(data[movienumber-1,1][0,0,:]))
    if highlim!=0 and highlim<msmax:
        msmax = highlim
    meanoverparticle = np.mean(np.mean(data[movienumber-1,1][x-particlexsize:x+particlexsize, y-particleysize:y+particleysize,20+sof:msmax-mof], axis=0), axis=0)
    
    fig, ax1 = plt.subplots(figsize=(10,4))
    color = 'tab:red'
    ax1.set_xlabel('time (frames)')
    ax1.set_ylabel('measurement', color=color)
    ax1.plot(meanoverparticle, color=color)
    ax1.tick_params(axis='arb.u.', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('signal', color=color)  # we already handled the x-label with ax1
    ax2.plot(signal_potential[20+mof:msmax-sof], color=color)
    ax2.tick_params(axis='volt', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("%s found offset (movie - signal): %d" %(interestingspots[lookat][0], moviesignaloffset))
    if save_automatically:
        figurename = interestingspots[lookat]['def']+"/measurement_and_signal"
        customSave(figurename, meanoverparticle)
    plt.show()
    
    
def spotvssignal(lookat, lowlim, highlim):
    global thing
    plt.figure(figsize=(10,10))
    x, y = interestingspots[lookat][1], interestingspots[lookat][2]
    x1, x2, y1, y2 = dfsm.getROIaroundXYinFigure(x, y, data[movienumber-1,1][:,:,0], xsize = particlexsize, ysize = particleysize)
    meanoverparticle = np.mean(np.mean(data[movienumber-1,1][x1:x2, y1:y2, 20+sof:msmax-mof], axis=0), axis=0)
    plt.plot(signal_potential[20+mof:msmax-sof], meanoverparticle, '.')
    thing = np.mean(meanoverparticle[:200*int(len(meanoverparticle)/200)].reshape(-1,200), axis=0)  # Average over the points.
    plt.plot(signal_potential[20+mof:int(20+mof+200)], thing, 'r-')
    plt.plot(signal_potential[[20+mof,int(20+mof+200)-1]], thing[[0,-1]], 'r-')  # Connect end to beginning
    plt.title("%s spot from frame %d to %d"%(interestingspots[lookat][0],lowlim,highlim))
    plt.xlabel('potential (V)')
    plt.ylabel('signal (arb.u.)')
    if save_automatically:
        figurename = interestingspots[lookat]['def']+"/measurement_vs_signal"
        customSave(figurename, np.array([thing,signal_potential[20+mof:int(20+mof+200)]]))
    plt.show()
    
    fig, ax1 = plt.subplots(figsize=(10,4))
    color = 'tab:red'
    ax1.set_xlabel('time (frames)')
    ax1.set_ylabel('measurement average', color=color)
    ax1.plot(np.reshape([thing,thing],-1), color=color)
    ax1.tick_params(axis='arb.u.', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('signal', color=color)  # we already handled the x-label with ax1
    ax2.plot(signal_potential[20+mof:20+len(np.reshape([thing,thing],-1))], color=color)
    ax2.tick_params(axis='volt', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("average total signal for %s spot"%(interestingspots[lookat][0]))
    if save_automatically:
        figurename = interestingspots[lookat]['def']+"/average_measurement_and_signal"
        customSave(figurename, thing)
    plt.show()
    
    
#%%
# Settings
# Configure settings, make folders, open data files and declare some functions:

# Configure:
folder = "/media/kevin/My Passport/2019-04-04-measurements/"
filename = "01_salt_triangle_1vpp"
movienumber = 1
Tframe = 1/200.  # In seconds the (estimated) time per frame 

savedirName = folder+filename+"_m"+str(movienumber-1)+"/"
save_automatically = True

extension = ".hdf5"



#%%
# Do things to the whole chosen movie:
getSignalperiod()
createFolders()

# Import data
data = dfsm.ImportHDF5data(folder+filename+extension)
data.setkey(0)
print("Wherein there is : ", data.getkeys())
data.resetkeys()

getvariablesandmiddleline()

getmeanIandvarI()

spotfinder_initial()

#%%
# Find interesting spots:
""" ITO-specle """
spotfinder(0, 0, 0)

#%%
# Find interesting spots:
""" High-variance_high-intensity """
spotfinder(1, 0, 0)

#%%
# Find interesting spots:
""" Low-variance_high-intensity """
spotfinder(2, 0, 0)

#%%
# Find interesting spots:
""" Dark-area_no-var-no-int """
spotfinder(3, 0, 0)

#%%
# Find interesting spots:
""" Random_spot """
spotfinder(4, 0, 0)

#%%
# Find interesting spots:
""" Random_spot2 """
spotfinder(5, 0, 0)


#%%
# Show all chosen spots:
spotfinder_final()
for x in interestingspots:
    print(x[0], x[1], x[2])

#%%
# Look at harmonic components using FFT transforms:

dcbase=5  # I put this higher than 5 to more easily filter more noise. 
maxFFTc=100
particlexsize = 10
particleysize = 5
harmonic_components()


#%%
# Plot the signal and measurement overlayed in one plot:

lookat = 1
lowlim = 0
highlim = 0


spotandsignal(lookat, lowlim, highlim)
spotvssignal(lookat, lowlim, highlim)

#%%
# Plot the signal versus the measurement:

for lookat in range(6):
    spotandsignal(lookat, lowlim, highlim)
    spotvssignal(lookat, lowlim, highlim)



print("DONE")







# With GNP positions:
##%%
## Find interesting spots:
#""" ITO-specle """
#spotfinder(0, 430, 25)
#
##%%
## Find interesting spots:
#""" High-variance_high-intensity """
#spotfinder(1, 225, 54)
#
##%%
## Find interesting spots:
#""" Low-variance_high-intensity """
#spotfinder(2, 229, 75)
#
##%%
## Find interesting spots:
#""" Dark-area_no-var-no-int """
#spotfinder(3, 0, 0)
#
##%%
## Find interesting spots:
#""" Random_spot """
#spotfinder(4, 20, 85)
#
##%%
## Find interesting spots:
#""" Random_spot2 """
#spotfinder(5, 633, 62)






