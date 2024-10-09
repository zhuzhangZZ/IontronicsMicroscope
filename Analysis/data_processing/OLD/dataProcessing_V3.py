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

#%%
# Settings
# Configure settings, make folders, open data files and declare some functions:

# Configure:
folder = "/media/kevin/My Passport/2019-02-18-measurements/"
filename = "w1-NaCl-Pt"
movienumber = 1
Tframe = 1/200.  # In seconds the time per frame 

hertz = 1
Tsignal = 1./hertz  # In seconds, used a bit
Asignal = 1.  # In volts, never used
savedirName = folder+filename+"_triangle_2V-1H/"

save_automatically = True
extension = ".hdf5"


# Create folders
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
# Open data:
data = dfsm.ImportHDF5data(folder+filename+extension)
data.setkey(0)
print("Wherein there is : ", data.getkeys())
data.resetkeys()
# Some functions that are easier to declare here:
def customSave(figurename, plotted, raw = True):
    # Needs to be called when the figure is in "memory" needs a logical name of the figure and the plotted np data.
    # Use to save the figure currently in use and the raw np data used to make it, puts it in the correct folder.
    if save_automatically: 
        if raw:
            previewfile = dirName+"/"+figurename+"_"+filename+'_rawImage.npy'
        np.save(previewfile, plotted)
        plt.savefig(dirName+"/"+figurename+"_"+filename)


#%% 
# Check start and end of data and plot intensity of middle line over time:

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
del middleline  # Do some management of data because it is close.
Nperiods = (endframe-startframe)/FPsignal  # This variable is used for knowing how many peaks to find for the self correlation
print("startframe: %d \nendframe: %d \nxrange: 0 to %d \nyrange: 0 to %d \nNperiods: %lf (%d frames per signal in %d frames)" %(startframe, endframe, xrange, yrange, Nperiods, FPsignal, endframe-startframe))


#%%
# Calculate time averaged intensity and variance/intensity of movie:

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
fig = plt.figure(figsize=(18,4)) 
dfsm.plotlogMean(mean_img)
if save_automatically:
    customSave("overview/Time-Averaged-Picture", mean_img)
plt.show()


#%%
# Plot of the variance of the picture: 

fig = plt.figure(figsize=(18,8))
dfsm.plotlogMeanlogVar(mean_img, var_img_MOD)
if save_automatically: 
    figurename = "overview/Variance"
    customSave(figurename, var_img)
plt.show()


#%%
# Find spot of interest, which is the spot with maximum variance: (with some limitations)

# Set what specle you look at:
lookat = 0
# First look at all specles if they are how you want them for this used field of view
# Only needs to change once per field of view:
find_spots_with_this_movie = False

# Manually edit if spots are not fixed yet, it outosets some things:
if find_spots_with_this_movie:
    binnedforfindingstuffvar = dfsm.doBin(dfsm.doBin(var_img_MOD))
    binnedforfindingstuffint = dfsm.doBin(dfsm.doBin(mean_img))
    hvarx, hvary = np.unravel_index(np.argmax(binnedforfindingstuffvar*binnedforfindingstuffint), binnedforfindingstuffvar.shape)
    lvarx, lvary = np.unravel_index(np.argmax(binnedforfindingstuffint**(0.5)/binnedforfindingstuffvar), binnedforfindingstuffvar.shape)
    dvarx, dvary = np.unravel_index(np.argmin(binnedforfindingstuffvar*binnedforfindingstuffint), binnedforfindingstuffvar.shape)
    kvarx, kvary = np.unravel_index(np.product([ord(l) for l in 'Kevin'])%mean_img.size, mean_img.shape)
    svarx, svary = np.unravel_index(np.product([ord(l) for l in 'Sanli'])%mean_img.size, mean_img.shape)
    
    interestingspots = np.array([('ITO-specle', 637, 87),
                                 ('High-variance_high-intensity', hvarx*4, hvary*4),
                                 ('Low-variance_high-intensity', lvarx*4, lvary*4),
                                 ('Dark-area_no-var-no-int', dvarx*4, dvary*4),
                                 ('Random_spot', kvarx, kvary),
                                 ('Random_spot2', svarx-32, svary-8)], dtype=[('def', np.unicode_, 100), ('x', int), ('y', int)])

# Make plot of area:
xmin, xmax, ymin, ymax = dfsm.getROIaroundXYinFigure(interestingspots[lookat]['x'], interestingspots[lookat]['y'], var_img)
fig = plt.figure(figsize=(9,4.5))
dfsm.plotlogMeanlogVarwithROI(binnedforfindingstuffint, binnedforfindingstuffvar, np.array(('binned_interestingspot', interestingspots[lookat]['x']/4, interestingspots[lookat]['y']/4), dtype=[('def', np.unicode_, 100), ('x', int), ('y', int)]))
plt.show()
fig = plt.figure(figsize=(18,9))
dfsm.plotlogMeanlogVarwithROI(mean_img, var_img_MOD, interestingspots[lookat])
plt.show()


#%%
# Look at harmonic components using FFT transforms:
for lookat in range(len(interestingspots)):
    
    maxFFTc = 100 # fourier index above which is irrelevant for the analysis
    dcbase = 5   # fourier index below which counts as drift
    
    # First make folders for the spots just identified:
    for f in interestingspots[:]['def']:
        if not os.path.exists(dirName+f):
            os.makedirs(dirName+f)
    # Do fast fourier transform on area around spot
    x, y = interestingspots[lookat]['x'], interestingspots[lookat]['y']
    xmin, xmax, ymin, ymax = dfsm.getROIaroundXYinFigure(x, y, movie[:,:,0], 15, 5)
    FFTspecROI = np.fft.rfft(movie[xmin:xmax,ymin:ymax], axis=2)/(safeendframe - safestartframe)
    # Find maximum valued frequency:
    mfreq = dcbase + np.argmax(abs(FFTspecROI[x-xmin, y-ymin, dcbase:maxFFTc]))
    shfreq = mfreq + 1 + np.argmax(abs(FFTspecROI[x-xmin, y-ymin, mfreq+1:maxFFTc]))
    thirdfreq = shfreq + 1  + np.argmax(abs(FFTspecROI[x-xmin, y-ymin, shfreq+1:maxFFTc]))
    # Plot fourier components:
    fig = plt.figure(figsize=(10,6))
    plt.plot(np.arange(dcbase, maxFFTc), np.log10(abs(FFTspecROI[x-xmin, y-ymin, dcbase:maxFFTc]))) # Give correct "x"axis
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
    fig = plt.figure(figsize=(18,12))
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

#%%
# Oscillations for SOI after summing around spot with highest harmonic component

lookat = 0
# Make true if you want to setup the signal based on this specle (usually ITO specle area)
setup_with_this_speckle = True

# Spot halfwidths:
wx, wy = 10, 5

# Plot raw intensity for spot around saved pixel:
mx, my = interestingspots[lookat]['x'], interestingspots[lookat]['y']
print("Center of spot at:", mx, my, interestingspots[lookat]['def'])
avg_spot = np.mean(np.mean(data[movienumber-1,1][mx-wx:mx+wx+1, my-wy:my+wy+1,startframe:endframe]-96, axis=0), axis=0)
fig = plt.figure(figsize=(18,6))
plt.plot(avg_spot,'.')
plt.title("Bright speckle intensity (specle around x=%d, y=%d)" %(mx, my))
plt.xlabel("Time (frames)")
plt.ylabel("Signal (arb.u.)")
if save_automatically: 
    plt.savefig(dirName+interestingspots[lookat]['def']+"/AveragedPeriod_SOI_"+filename)
plt.show()


#%%
# Look at self correlation (not that usefull but we can so why not)

FFTspecFiltered = FFTspecROI[x-xmin, y-ymin, :].copy()
FFTspecFiltered[:dcbase] = 0  # Use same filter as before
FFTspecFiltered[maxFFTc:] = 0  # Use same filter as before
specFiltered = np.fft.irfft(FFTspecFiltered)
# First show specle in time after filtering:
fig = plt.figure(figsize=(8,6))
plt.plot(specFiltered)
plt.title("Spot of interest after filtering some Fourier components")
plt.xlabel("Time (frames)")
plt.ylabel("Intensity (arb. u.)")
if save_automatically:
    figurename = interestingspots[lookat]['def']+"/Intensity_SOI-postFFT"
    customSave(figurename, specFiltered)
plt.show()
# Trow away the sides of the data because the FFT filter makes them ugly: (Throwing away 30% in total right now)
specFilteredclean = specFiltered[int(0.15*len(specFiltered)):-int(0.15*len(specFiltered))]  
correlationAfterFramesFFT = dfsm.timeDependentCorrelation(specFilteredclean)
fig = plt.figure(figsize=(8,6))
plt.plot(correlationAfterFramesFFT)
plt.title("Self correlation of spot of interest after\n filtering some Fourier components")
plt.xlabel("Time (frames)")
plt.ylabel("Correlation")
if save_automatically:
    figurename = interestingspots[lookat]['def']+"/Self-Correlation-SOI-postFFT"
    customSave(figurename, correlationAfterFramesFFT)
plt.show()
print("First peak in selfcorrelation:", int(FPsignal/2)+np.argmax(correlationAfterFramesFFT[int(FPsignal/2):int(FPsignal*3/2)]))


#%%
# Guess a triangle wave to triangle esque response for making Intensity voltage plot:

TWphase = 0 # Guess phase
nicebitstart = 1500
nicebitend = 3000

avg_spot_filtered = dfsm.pixelFFTFilter(avg_spot, dcbase, maxFFTc)[nicebitstart:nicebitend]
TWamplitude = 0.8*np.max(avg_spot_filtered)
TWfrequency = 1/(FPsignal)
# Define triangle wave function:
def triangleWave(twopift):
    if np.size(twopift) <= 1:
        if twopift%(2*np.pi)<np.pi: 
            sign = 1 
        else: 
            sign = -1
        loc = twopift%np.pi
        if(loc<np.pi/2):
            return sign*loc*2/np.pi
        else:
            return sign*(2-loc*2/np.pi)
    else: 
        return np.array([triangleWave(twopift[i]) for i in range(len(twopift))])
# Plot filtered signal and first guess at signal:
if setup_with_this_speckle:
    plt.figure(figsize=(12,6))
    plt.plot(avg_spot_filtered, '.')
    xx = np.arange(len(avg_spot_filtered))
    signal = Asignal*triangleWave(2*np.pi*TWfrequency*(xx-TWphase))
    plt.plot(TWamplitude/Asignal*signal)
    plt.show()


#%%
# Fit or use fitted trianglefunction

if setup_with_this_speckle:
    # Switch between minimum and maximum to base frequency on:
    use_max_for_freq = True
    
    # Frequency:
    if use_max_for_freq:
        start = np.argmax(avg_spot_filtered[:int(1/TWfrequency)])
    else:
        start = np.argmin(avg_spot_filtered[:int(1/TWfrequency)])
    lookrange1 = int(2/TWfrequency/3)
    lookrange2 = int(4/TWfrequency/3)
    periodlist = []
    while(start+lookrange2 < len(avg_spot_filtered)):
        if use_max_for_freq:
            nexts = np.argmax(avg_spot_filtered[start+lookrange1:start+lookrange2])
        else:
            nexts = np.argmin(avg_spot_filtered[start+lookrange1:start+lookrange2])
        periodlist.append(nexts+lookrange1)
        start = start+nexts+lookrange1
    TWfrequency2 = 1/np.mean(periodlist)
    # Phase:
    startlist = [sum(periodlist[:i]) for i in range(len(periodlist))]+np.argmin(avg_spot_filtered[:int(1/TWfrequency)])
    startlist = np.array([startlist[i]%(1/TWfrequency2) for i in range(len(startlist))])
    startlist[startlist>TWfrequency2/2] -= 1/TWfrequency2
    TWphase2 = np.mean([startlist[i]%(1/TWfrequency2) for i in range(len(startlist))])
    if use_max_for_freq:
        TWphase2 = TWphase2 - 1/4*(1/TWfrequency2)
    else:
        TWphase2 = TWphase2 - 3/4*(1/TWfrequency2)
    # Report findings
    print("found freq", TWfrequency2)
    print("found phase", TWphase2)
# Plot fitted signal and data:
plt.figure(figsize=(12,6))
plt.plot(avg_spot_filtered, '.')
xx = np.arange(len(avg_spot_filtered))
signal = Asignal*triangleWave(2*np.pi*TWfrequency2*(xx-TWphase2))
plt.plot(TWamplitude/Asignal*signal)
if setup_with_this_speckle:
    for i in range(len(periodlist)):
        plt.vlines(int(np.argmax(avg_spot_filtered[:int(1/TWfrequency)])+sum(periodlist[:i])), -TWamplitude, TWamplitude)
plt.title("Intensity and recreated signal for spot around x=%d, y=%d" %(mx, my))
plt.xlabel("Time (frames)")
plt.ylabel("Intensity (arb. u.)")
if save_automatically:
    figurename = interestingspots[lookat]['def']+"/Spot_intensity-and-signal"
    customSave(figurename, np.transpose([signal, avg_spot_filtered]))
plt.show()

#%%
# Plot Intensity vs the signal

plt.figure(figsize=(12,10))


plt.plot(signal, avg_spot_filtered, '.', color='grey')

len_zerofilled_filtered2 = int(FPsignal-len(avg_spot_filtered)%FPsignal)
avg_spot_filtered2 = np.append(avg_spot_filtered, np.array([0]*len_zerofilled_filtered2))
avg_spot_filtered2 = avg_spot_filtered2.reshape(-1, int(FPsignal))
avg_spot_filtered2 = np.append(np.mean(avg_spot_filtered2[:,:-len_zerofilled_filtered2], axis=0),np.mean(avg_spot_filtered2[:-1,-len_zerofilled_filtered2:], axis=0))

plt.plot(signal[0:int(FPsignal)], avg_spot_filtered2, '.r')

plt.title("Intensity vs signal for spot around x=%d, y=%d" %(mx, my))
plt.xlabel("Signal")
plt.ylabel("Intensity")
if save_automatically:
    figurename = interestingspots[lookat]['def']+"/Spot_average-intensity-vs-signal"
    customSave(figurename, np.transpose([signal[0:int((1/TWfrequency2))], avg_spot_filtered2]))
plt.show()


#%%
# Save all interesting data:

metadataishthing = np.array([("movienumber", movienumber), 
                             ("camera_fps", Tframe), 
                             ("frames_per_signal", FPsignal), 
                             ("signal_amplitude_V", Asignal), 
                             ("signal_period_seconds", Tsignal), 
                             ("n_frames", nf), 
                             ("interestingspot_0_def_"+interestingspots[0]['def'], 0.),
                             ("interestingspot_0_x", interestingspots[0]['x']),
                             ("interestingspot_0_y", interestingspots[0]['y']), 
                             ("interestingspot_1_def_"+interestingspots[1]['def'], 1.),
                             ("interestingspot_1_x", interestingspots[1]['x']),
                             ("interestingspot_1_y", interestingspots[1]['y']), 
                             ("interestingspot_2_def_"+interestingspots[2]['def'], 2.),
                             ("interestingspot_2_x", interestingspots[2]['x']),
                             ("interestingspot_2_y", interestingspots[2]['y']), 
                             ("interestingspot_3_def_"+interestingspots[3]['def'], 3.),
                             ("interestingspot_3_x", interestingspots[3]['x']),
                             ("interestingspot_3_y", interestingspots[3]['y']), 
                             ("interestingspot_4_def_"+interestingspots[4]['def'], 4.),
                             ("interestingspot_4_x", interestingspots[4]['x']),
                             ("interestingspot_4_y", interestingspots[4]['y']),
                             ("interestingspot_5_def_"+interestingspots[5]['def'], 5.),
                             ("interestingspot_5_x", interestingspots[5]['x']),
                             ("interestingspot_5_y", interestingspots[5]['y']) ], dtype='object, double')


metadataishthingfile = dirName+"/data_"+filename+'.npy'
np.save(metadataishthingfile, metadataishthing)

# Also finally save plot with all spots of interest in it:
fig = plt.figure(figsize=(18,5))
ax = plt.imshow(np.log10(np.transpose(mean_img)), cmap='Greys_r')
plt.title("Logarithm of Averaged Intensity")
plt.ylabel("y")
plt.xlabel("x")
clb = plt.colorbar()
clb.set_label('(arb.u.)')
for spot in interestingspots:
    plt.scatter(spot['x'], spot['y'], s=10, marker='o', label=spot['def'])
plt.legend()
# Put a legend below current axis
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
          fancybox=True, shadow=True, ncol=len(interestingspots))
if save_automatically:
    figurename = "overview/Spot-locations"
    customSave(figurename, interestingspots)
plt.show()






