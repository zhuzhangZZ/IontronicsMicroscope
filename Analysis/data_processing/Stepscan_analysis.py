# -*- coding: utf-8 -*-
"""
_.py 
Analyses _.
Allows selection of a certain 

@author: Kevin Namink <k.w.namink@uu.nl>

Feedback and Comments by Sanli Faez
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# To import our library with functions you might need to put 
# the DFSM.py file in the same folder as this file
import DFSM as dfsm


# Settings
# Usage: 
# Change folder, filename and framerate for the desired movie.
# It can be usefull to add some information to the info variable.


info = """
Put info here, like from the readme file associated with the measurement being
analysed, so it is always also available.
"""

# Configure:
folder = "/media/kevin/My Passport/2019-12-05-extremestepscan/"
filename = "Measurement3"
movienumber = 0
extension = ".hdf5"
framerate = 200.  # In frames per second

#%% 
# Import and handle signal data
# noise_factor is a setting that helps select the edges of the different offsets.
# Values usually lie between 1. and 5. 
# The resulting amount of signals (Nsignals that is displayed) should be close to what you expect.
noise_factor=1
# TLDR: run block, check dfsm for how it works.

# Import DAC data:  ###########################################################
DACdata = np.load(folder+filename+"_m"+str(movienumber)+".npy")
DAC_v, DAC_c, DAC_led = DACdata[:,0], DACdata[:,1], DACdata[:,2]
# Handle DAC data to get the signal for every applied potential.
# When there was only one type applied it will be the only one present
signals = \
    dfsm.DAC_signal_properties(DAC_v, varied_offset=True, noise_factor=noise_factor)
# Every signal contains the following things in a dictionary:
''' "period", "range", "phase", "amplitude", "offset", "shape" '''

# Imports camera data: ########################################################
data = dfsm.ImportHDF5data(folder+filename+extension)
# Find some properties of the movie: 
xrange, yrange, startframe, endframe, quicklook = \
    dfsm.movie_properties(data, movienumber)
# Plot quicklook, which is useful for estimating movie quality:
plt.plot(quicklook[startframe:endframe])
plt.title("Average of the center line")
plt.xlabel("Frame number")
plt.ylabel("Intensity (arb.u.)")
plt.show()

# Report found values: ########################################################
Nperiods = (endframe-startframe)/signals[0]['period']
Tframe = 1/framerate
print("startframe: %d \nendframe: %d \nxrange: 0 to %d    yrange: 0 to %d \
      \nNsignals: %d \nNperiods: %lf (%lf frames per signal in %d frames)"
      %(startframe, endframe, xrange, yrange, len(signals),
        Nperiods, signals[0]['period'], endframe-startframe))
# Find a rough mean image:
mean_img = np.mean(data[movienumber,1][:,:,
                   startframe:endframe:int((endframe-startframe)/10)], axis=2)

# Find offset between movie and signal: #######################################
moviesignaloffset = dfsm.movie_signal_offset(quicklook, DAC_led)

# Overwrite manually if needed:
#moviesignaloffset = 6

# Print found value:
print("Found offset (movie - signal):", moviesignaloffset)
# Apply found offset by calculating movie and signal offsets:
mof, sof, msmax = \
    dfsm.create_offset_parameters(moviesignaloffset, quicklook, DAC_led)
# Plot result:
plt.figure(figsize=(8,4)) ; plt.subplot(121)
plt.plot(quicklook[0:60]/max(quicklook[mof:60-sof]), label="unedited movie")
plt.plot(DAC_led[0:60]/max(DAC_led[sof:60-mof]), label="unedited signal")
plt.title("alignment between signal and movie") ; plt.legend() ; plt.subplot(122)
plt.plot(quicklook[mof:60-sof]/max(quicklook[mof:60-sof]), label="edited movie")
plt.plot(DAC_led[sof:60-mof]/max(DAC_led[sof:60-mof]), label="edited signal")
plt.title("alignment between signal and movie") ; plt.legend() ; plt.show()


#%%
# Select a spot
# TLDR: set x_guess and y_guess and execute block until satisfied

# Usage: 
# Guess the center pixel of the spot you are interested in.
# The found spot location is plotted.
# This step can also indicate when particles are oversaturated or 
# otherwise compromised. (the found intensity will be 32767.0)

# What it does, 3 options: (Option 2 is recommended)
#
# 1. It looks in an area 10 by 10 pixels around your guess for the peak intensity 
# in the "mean_img" image. 
# Then it finds the FWHM around this peak in the x and y direction. 
# The rectangle found by these FWHM values around the maximum is used as the 
# particle, so is summed over when looking for the particle intensity.
#
# 2. It looks in an area 10 by 10 pixels around your guess for the peak intensity 
# in the "mean_img" image. 
# The height and width variables then represent the amount of pixels to expand around the found peak
# It goes both up and down by the height variable. Same for width.
# Recommended values are 1 and 1, this way you get some averaging but little chance of strange extra signals.
#
# 3. Look at a region around the guessed spot. (Activated by making do_a_region = True)
# The height and width variables then represent the amount of pixels to expand around the guess
# It goes both up and down by the height variable. Same for width.

# You can also write a new method of your own.

# Select a spots (guess its center):
x_guess = 70
y_guess = 15

# Only for option 2 and 3:
height  = 1
width   = 1

# Select method you want:
particle_finding_option_chosen = 2


########################################
# Find the particle:
if particle_finding_option_chosen < 3:
    # Calculate local maximum intensity:
    search_area_size = 10  # You can change the search area
    x_found, y_found = \
        dfsm.find_max_I_around_guess(mean_img, x_guess, y_guess, areasize=search_area_size)
else: 
    # Find center of region and find region:
    x_found = x_guess
    y_found = y_guess
if particle_finding_option_chosen == 1:
    # Find FWHM: 
    xx, x_range, x1, yy, y_range, y1 = \
        dfsm.find_FWHM_for_particle(mean_img, x_found, y_found, x_search=30, y_search=16, plot_it=True)
else:
    x1, x2, y1, y2 = dfsm.getROIaroundXYinFigure(x_found, y_found, mean_img, xsize=width, ysize=height)
    x_range, y_range = (x1, x2), (y1, y2)
print("x_found:", x_found, " and y_found:", y_found)
print("x_range:", x_range, " and y_range:", y_range)
print("With intensity", mean_img[x_found,y_found])
if mean_img[x_found,y_found] == 32767.0:
    print("Found particle is (probably) oversaturated.")

########################################
# Plot mean_img with particle location:
plt.figure(figsize=(16,5))
plt.imshow(np.log10(np.transpose(mean_img)))
plt.scatter(x_guess, y_guess, s=10, color="xkcd:pink", label = "guess: I=%d"%mean_img[x_guess,y_guess])
plt.scatter(x_found, y_found, s=10, color="xkcd:red", label = "found: I=%d"%mean_img[x_found,y_found])
plt.title("Logarithm of Averaged Intensity")
plt.xlabel("x") ; plt.ylabel("y") ; plt.legend()
plt.show()


#%%
# Process intensity data:
# TLDR: run it and look at figures

# Usage: 
# Every step is commented on what it does, for more information: see dfsm
# Resulting data is saved as "results"
mobile_peak = True

results = []
for num, signal in enumerate(signals):
    if np.diff(signal['range']) < 100:
        continue
    if signal['period'] < 10: 
        continue
    
    if mobile_peak:  # Adjust peak location to correct for drift.
            temp_mean_img = np.mean(data[movienumber,1][:,:,signal['range'][0]:signal['range'][0]+10], axis=2)
            x_mobile, y_mobile = \
                dfsm.find_max_I_around_guess(temp_mean_img, x_found, y_found, areasize=5)
            x1, x2, y1, y2 = dfsm.getROIaroundXYinFigure(x_found, y_found, temp_mean_img, xsize=width, ysize=height)
            x_range, y_range = (x1, x2), (y1, y2)
    
    # Equilibriate for some periods (since at least 5 periods are expected in earlier functions, 4 is safe but maybe not enough)
    equilibriate_skip = signal["period"]*4
    # Sum over particle adjusted for overlapping and skipping LED flash/leaving enough time to equilibriate:
    particle = np.sum(np.sum(data[movienumber,1][x_range[0]:x_range[1], y_range[0]:y_range[1],int(signal["range"][0]+equilibriate_skip+mof):int(signal["range"][1]-sof)] - 96, axis=0), axis=0)
    
    # Sometimes there is a 2 frame period offset from yet unknown reason.
    # Fix with the following:
    particle = (np.roll(particle,-1)+particle)/2
    
    # Adjust DAC data for overlapping and skip the LED flash:
    adjusted_signal = DAC_v[int(signal["range"][0]+equilibriate_skip+sof):int(signal["range"][1]-mof)]
    adjusted_current = DAC_c[int(signal["range"][0]+equilibriate_skip+sof):int(signal["range"][1]-mof)]
    
    # Correct particle drift by either fitting a polynominal or using a FFT filter
    # options: polyorfft=['fft', 'poly'], polyorder=8 and fftfilter=16
    # Notes: FFT can do more extreme corrections if they are needed, poly is smoother.
    driftcorrected_particle = \
        dfsm.correct_particle_drift(particle, polyorfft='poly', polyorder=12, fftfilter=16)
    
    # Start a plot to put information on the drift corrrection and cycle averaging in:
    fig = plt.figure(figsize=(12,6))
    plt.plot(particle, color="xkcd:orange", label="signal before correction")
    plt.plot(driftcorrected_particle, color='xkcd:green', label="signal after correction")
    # Ratio of points allowed in a single cycle to be outside the whole measurements' gaussian 5% distance
    outlier_rate = 0.2  # 0.2 should only filter truely bad parts.  
    # Calculate cycle averages while filling plot:
    i, v, c = \
        dfsm.cycleaverage_particle_and_DAC(driftcorrected_particle, 
                                           adjusted_signal, adjusted_current, 
                                           signal["period"], signal["phase"], 
                                           outlier_rate)
    # Finish plot:
    plt.title("Drift correction and accepted cycles", size=16)
    plt.xlabel("frames since signal start", size=14)
    plt.ylabel("intensity (counts)", size=14)
    plt.legend() ; plt.show()
    
    # Change start of data to logical position:
    roll_amount = - np.argmax(v) # + int(1/4*signal["period"])
    i = np.roll(i, roll_amount) ; v = np.roll(v, roll_amount) 
    c = np.roll(c, roll_amount) ; t_arr = Tframe*np.arange(len(i))
    
    results.append({"cycle_intensity": i, "cycle_potential": v, "cycle_current": c,
                    "potential": adjusted_signal, "intensity": driftcorrected_particle, 
                    "x_range": x_range, "y_range":y_range})
    
    # Plot processed intensity data: ##########################################
    # Usage: plots stuff, edit to your wish
    ###########################################################################
    # Show intensity and potential:
    fig = plt.figure(figsize=(10,4)) ; ax1 = fig.add_subplot(1,1,1)
    color = 'tab:red' ; ax1.set_xlabel('time ($s$)', size=16)
    ax1.set_ylabel('$\mathrm{\Phi}_{\mathrm{cell}}$ ($V$)', size=16, color=color) 
    ax1.plot(t_arr, v, '.', color=color) ; ax1.tick_params('y', colors=color)
    ax2 = ax1.twinx() ; color = 'xkcd:royal blue'
    ax2.set_ylabel('I/I$_{avr}$ ', size=16, color=color)
    ax2.plot(t_arr, i/np.mean(i), '.', color=color)
    ax2.tick_params('y', colors=color) ; plt.show()
    
    # Show intensity vs potential #############################################
    separateupdown = [np.argmax(v), np.argmin(v)] ; half = int(len(i)/2 + 1)
    fig = plt.figure(figsize=(10,6)) ; ax1 = fig.add_subplot(1,1,1)
    # Uncomment to show every single point:
    #ax1.plot(adjusted_signal, driftcorrected_particle/np.mean(i), '.', markersize=2, color='xkcd:grey', label="All cycles") 
    ax1.set_xlabel('$\Phi_{cell}$ (V)', size=18) ; ax1.set_ylabel('$I/I_{avr}$', size=18)
    ax1.plot(np.roll(v, separateupdown[0])[:half], np.roll(i, separateupdown[0])[:half]/np.mean(i), 
             '.-b', markersize=15, label="decreasing potential")
    ax1.plot(np.roll(v, separateupdown[1])[:half], np.roll(i, separateupdown[1])[:half]/np.mean(i), 
             '.-r', markersize=15, label="increasing potential")
    # You can apply a Savitzky-Golay filter to an array to make it smoother.
    # This filter fits polynominals to part of the data to effectively average it. 
    if False: # make and plot an smoothed curve
        part_s = int(len(i)/10) + 1 - int(len(i)/10)%2  # get an always odd number that is approximately 1/10th of the size of the array
        if part_s<5: part_s = 5  # minimal value
        ia = savgol_filter(i, part_s, 3, mode='wrap', deriv=0)
        ax1.plot(np.roll(v, separateupdown[0])[:half], np.roll(ia, separateupdown[0])[:half]/np.mean(i), '-', linewidth=4, color='xkcd:dark blue')
        ax1.plot(np.roll(v, separateupdown[1])[:half], np.roll(ia, separateupdown[1])[:half]/np.mean(i), '-', linewidth=4, color='xkcd:dark red')
    # Finish plot:
    ax1.tick_params(labelsize=14) ; plt.legend() ; plt.show()
    print("For offset:", signal["offset"], "   progress:", num/len(signals)*100,"percent")


#%%
# For when doing a varied offset measurement: get the PDOC proportionality depending on the offset

# Fit a 1st order polynominal to the normalized intensity for each result
fits = []
for it, bit in enumerate(results):
    if len(bit['cycle_potential']) < 3:
        continue
    if (bit['cycle_potential'] != [ float(b) for b in bit['cycle_potential'] ]).any():
        continue
    fit = np.polyfit(bit['cycle_potential'], bit['cycle_intensity']/np.mean(bit['cycle_intensity']), 1)
    if (fit != [ float(b) for b in fit ]).any():
        continue
    fits.append(np.append(fit, signals[it]['offset']))
    
# Plot last result to see how it went:
    plt.figure()
    plt.plot(bit['cycle_potential'], bit['cycle_intensity']/np.mean(bit['cycle_intensity']))
    plt.plot(bit['cycle_potential'], bit['cycle_potential']*fit[0] + fit[1])
    plt.xlabel('potential')
    plt.ylabel('intensity and proportionality fit')
    plt.show()
    
npfits = np.array(fits)
np.save('/home/kevin/Documents/PDSM_data/2019-12-05-extremestepscan/'+filename+'_NPDOCvsOFFSET.npy', npfits)

#%%

# Plot:
plt.figure()
plt.plot(npfits[:,2], npfits[:,0], '.b')

ffit = np.polyfit(npfits[:,2], npfits[:,0], 3)
xx = np.linspace(np.min(npfits[:,2]), np.max(npfits[:,2]))
pp = np.poly1d(ffit)
plt.plot(xx, pp(xx), '-', linewidth=4, color='xkcd:orange')

plt.ylabel('PDOC proportionality')
plt.xlabel('offset /V')
plt.show()


# Load all data:
yy1 = np.load('/home/kevin/Documents/PDSM_data/2019-12-05-extremestepscan/Measurement1_NPDOCvsOFFSET.npy')
yy2 = np.load('/home/kevin/Documents/PDSM_data/2019-12-05-extremestepscan/Measurement2_NPDOCvsOFFSET.npy')
yy3 = np.load('/home/kevin/Documents/PDSM_data/2019-12-05-extremestepscan/Measurement3_NPDOCvsOFFSET.npy')

yy = np.append(yy1, np.append(yy2, yy3))
yy = yy.reshape(-1, 3)

# Fit to all data:
fffit = np.polyfit(yy[:,2], yy[:,0], 6)
xxx = np.linspace(np.min(yy[:,2]), np.max(yy[:,2]))
ppp = np.poly1d(fffit)

# One of the final plots:

SIZE = 16

plt.figure(figsize=(8,6))
plt.plot(yy[:,2], yy[:,0], '.b')
plt.plot(xxx, ppp(xxx), '-', linewidth=4, color='xkcd:orange')
plt.ylabel('PDOC angle', size=SIZE)
plt.xlabel('offset /V', size=SIZE)
plt.tick_params(labelsize=SIZE-1)
plt.grid()
plt.show()


#%%

plt.figure(figsize=(8,6))
integrated = np.cumsum(ppp(xxx))*np.diff(xxx[0:2])
# Normalization options: 
#  distance in xxx  ->  *np.diff(xxx[0:2])
#  range of appliedwaves  ->  *0.5
integrated = integrated - np.mean(integrated) + 1
plt.plot(xxx, integrated, '-', linewidth=4, color='xkcd:red')
plt.ylabel('PDOC (from angle)', size=SIZE)
plt.xlabel('potential /V', size=SIZE)
plt.tick_params(labelsize=SIZE-1)
plt.grid()
plt.show()


#%%
# Another two:

plt.figure(figsize=(8,6))
plt.plot(np.arange(1900,2200)/framerate, DAC_v[1900:2200], '-b')
plt.ylabel('potential /V', size=SIZE)
plt.xlabel('time /s', size=SIZE)
plt.xlim((1900/framerate,2198/framerate))
plt.tick_params(labelsize=SIZE-1)
plt.show()

plt.figure(figsize=(8,6))
plt.plot(np.arange(len(DAC_v))/framerate, DAC_v, '-b', linewidth=0.1)
plt.ylabel('potential /V', size=SIZE)
plt.xlabel('time /s', size=SIZE)
plt.xlim((0/framerate,len(DAC_v)/framerate))
plt.tick_params(labelsize=SIZE-1)
plt.show()

#%%
# Last one?


v_avrtot = [np.mean(r['cycle_potential']) for r in results]
i_avrtot = [np.mean(r['cycle_intensity']) for r in results]

plt.plot(v_avrtot, i_avrtot)
plt.show()

plt.plot(i_avrtot)
plt.show()




