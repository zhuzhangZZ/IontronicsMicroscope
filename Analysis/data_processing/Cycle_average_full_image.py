# -*- coding: utf-8 -*-
"""
_.py 
Analyses _.
It makes the cycle average for the full image and then analyses it by taking 
the angle of the intensity versus potential and plotting that in various ways.

Works best for NOT varied offset measurements, if you want to use it for that 
it needs editing.

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
Put some nice info here for your measurement
"""

# Configure:
folder = "/media/kevin/My Passport/20190723_Crgrid/"
filename = "Crgrid_cell_Nabr10mM_t2"
movienumber = 2
extension = ".hdf5"
framerate = 200.  # In frames per second
varied_offset = False  # If true there might be some edits required.

# Saving the result is quite usefull so it will always do so.
import os 
dir_path = os.path.dirname(os.path.realpath('__file__'))


#%% 
# Import and handle signal data
# TLDR: run block, check dfsm for how it works.

# Import DAC data:  ###########################################################
DACdata = np.load(folder+filename+"_m"+str(movienumber)+".npy")
DAC_v, DAC_c, DAC_led = DACdata[:,0], DACdata[:,1], DACdata[:,2]
# Handle DAC data to get the signal for every applied potential.
# When there was only one type applied it will be the only one present
signals = \
    dfsm.DAC_signal_properties(DAC_v, varied_offset=varied_offset, noise_factor=3)
# Every signal contains the following things in a dictionary:
# "period", "range", "phase", "amplitude", "offset", "shape"

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
print("startframe: %d \nendframe: %d \nxrange: 0 to %d \nyrange: 0 to %d \
      \nNperiods: %lf (%lf frames per signal in %d frames)"
      %(startframe, endframe, xrange, yrange, 
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

# Show how the mean image looks:
plt.figure(figsize=(16,5))
plt.imshow(np.log10(np.transpose(mean_img)))
plt.title("Logarithm of Averaged Intensity")
plt.xlabel("x") ; plt.ylabel("y") ; plt.legend()
plt.show()


#%%
# Process intensity data:
# This version finds the averaged cycle for each individual pixel

signal = signals[0] # for compatability with varied offset measurements, in varied offset measurements other signals can be used.

# Equilibriate for some periods (since at least 5 periods are expected in earlier functions, 4 is safe but maybe not enough)
equilibriate_skip = signal["period"]*4


# Adjust DAC data for overlapping and skip the LED flash:
adjusted_signal = DAC_v[int(signal["range"][0]+equilibriate_skip+sof):int(signal["range"][1]-mof)]
adjusted_current = DAC_c[int(signal["range"][0]+equilibriate_skip+sof):int(signal["range"][1]-mof)]
# Timerange for movie to use:
mrange1, mrange2 = int(signal["range"][0]+equilibriate_skip+mof), int(signal["range"][1]-sof)


# Calculate driftcorrection using only the quicklook data, it is not as good as otherwise possible but saves time and is still quite good.
polyorder = 12
driftcorrection = np.polyval(np.polyfit(np.arange(mrange2-mrange1), quicklook[mrange1:mrange2]-np.mean(quicklook[mrange1:mrange2]), polyorder), np.arange(mrange2-mrange1))

################ Start cycle averaging ########################################

# Parameters for cycle averaging:
signal_length = len(adjusted_signal)
number_of_cycles = int((signal_length - signal['offset'])/signal['period'])
number_of_accepted_cycles = 0
cycle_length = int(signal['period'])

# Things where the result of the cycle averaging will gather in:
result_intensity = np.zeros([xrange, yrange, int(signal['period'])])
result_signal = np.zeros(int(signal['period']))
result_current = np.zeros(int(signal['period']))

# Add the contribution of each cycle to the result:
for i in range(number_of_cycles):
    cycle_start = int(i*signal['period'] + signal['offset'])
    
    cycle_movie = data[movienumber,1][:,:,mrange1+cycle_start:mrange1+cycle_start+cycle_length] - 96
    cycle_signal = adjusted_signal[cycle_start:cycle_start + cycle_length]
    cycle_current = adjusted_current[cycle_start:cycle_start + cycle_length]

    result_intensity = result_intensity + cycle_movie/number_of_cycles
    result_signal = result_signal + cycle_signal/number_of_cycles
    result_current = result_current + cycle_current/number_of_cycles
    
    print(100*(i+1)/number_of_cycles, "percent done")

v = result_signal # for compatability with other pieces of analysis files
c = result_current # for compatability with other pieces of analysis files

# Change start of data to logical position:
roll_amount = - np.argmax(v) # + int(1/4*signal["period"])

result_intensity = np.roll(result_intensity, roll_amount, axis=2)
v = np.roll(v, roll_amount) 
c = np.roll(c, roll_amount)
t_arr = Tframe*np.arange(len(v))

# Save result as np file with dictionary:
np.save(dir_path+"/averaged_to1period_%s_m%d"%(filename, movienumber), {"intensity":result_intensity, 
        "mean_img":mean_img, "potential":v, "current":c, "t_arr":t_arr, 
        "info":"Averaged value for single period intensity and stuff for %s_m%d"%(filename, movienumber)})

i = result_intensity[2,3,:] # for compatability with other pieces of analysis files, instead of 2, 3 use whatever pixel you are interested in

#%%
# Look at one certain result, can be skipped when you are not interested.
# Not yet changed to only use saved data to become faster to run again but that is possible.

# Select a spots (guess its center):
x_guess = 390
y_guess = 60

# Size of region of interest:
height  = 1
width   = 1

search_area_size = 10  # You can change the search area
x_found, y_found = \
    dfsm.find_max_I_around_guess(mean_img, x_guess, y_guess, areasize=search_area_size)
x1, x2, y1, y2 = dfsm.getROIaroundXYinFigure(x_found, y_found, mean_img, xsize=width, ysize=height)
x_range, y_range = (x1, x2), (y1, y2)

########################################
# Plot mean_img with particle location:
plt.figure(figsize=(16,5))
plt.imshow(np.log10(np.transpose(mean_img)))
plt.scatter(x_guess, y_guess, s=10, color="xkcd:pink", label = "guess: I=%d"%mean_img[x_guess,y_guess])
plt.scatter(x_found, y_found, s=10, color="xkcd:red", label = "found: I=%d"%mean_img[x_found,y_found])
plt.title("Logarithm of Averaged Intensity")
plt.xlabel("x") ; plt.ylabel("y") ; plt.legend()
plt.show()


# Plot the PDOC of the location chosen:
i = np.mean(result_intensity[x1:x2,y1:y2,:], axis=(0,1))

# Show intensity and potential: ###############################################
fig = plt.figure(figsize=(10,4)) ; ax1 = fig.add_subplot(1,1,1)
color = 'tab:red' ; ax1.set_xlabel('time ($s$)', size=16)
ax1.set_ylabel('$\mathrm{\Phi}_{\mathrm{cell}}$ ($V$)', size=16, color=color) 
ax1.plot(t_arr, v, '.', color=color) ; ax1.tick_params('y', colors=color)
ax2 = ax1.twinx() ; color = 'xkcd:royal blue'
ax2.set_ylabel('I/I$_{avr}$ ', size=16, color=color)
ax2.plot(t_arr, i/np.mean(i), '.', color=color)
ax2.tick_params('y', colors=color) ; plt.show()
# Show intensity vs potential #################################################
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



#%%
# Plot angle for each pixel (can use a previously processed movie that was saved)

result = np.load(dir_path+"/averaged_to1period_%s_m%d.npy"%(filename, movienumber), allow_pickle=True).item()
xrange, yrange = result["mean_img"].shape

# Fit a 1rst order polynominal to the normalized intensity for each result
result_fits = np.zeros([xrange, yrange])
for xx in range(xrange):
    for yy in range(yrange):
        fit = np.polyfit(result['potential'], result['intensity'][xx,yy,:]/np.mean(result['intensity'][xx,yy,:]), 1)
        result_fits[xx, yy] = fit[0]

#%%
# Make various plots of this.


plt.figure(figsize=(12,3))
plt.imshow(mean_img.T, cmap='Greys')
plt.colorbar() ; plt.title("mean image")
plt.show()

plt.figure(figsize=(12,3))
plt.imshow(dfsm.doBin(result_fits).T, cmap='jet')
plt.colorbar() ; plt.title("normalized angle")
plt.show()

plt.figure(figsize=(12,3))
plt.imshow(dfsm.doBin(result_fits*result["mean_img"]).T, cmap='jet')
plt.colorbar() ; plt.title("real angle")
plt.show()

result_edit = np.copy(result_fits)
result_edit[mean_img<10000] = 0

plt.figure(figsize=(12,3))
plt.imshow(dfsm.doBin(result_edit).T, cmap='jet')
plt.colorbar() ; plt.title("real angle for only high intensity points")
plt.show()


#%%






