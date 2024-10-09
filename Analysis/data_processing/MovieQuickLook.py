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
folder = "C:\\Data\\Kevin\\UUTrack\\2019-06-05\\"
filename = "01-dodecane"
movienumber = 2
extension = ".hdf5"
framerate = 200.  # In frames per second
Tframe = 1/framerate

#%%
# Import:

# DAC data:
DACdata = np.load(folder+filename+"_m"+str(movienumber)+".npy")
DAC_v, DAC_c, DAC_led = DACdata[:,0], DACdata[:,1], DACdata[:,2]
signals = \
    dfsm.DAC_signal_properties(DAC_v)
# Every signal contains the following things in a dictionary:
# "period", "range", "phase", "amplitude", "offset", "shape"

# Camera data:
data = dfsm.ImportHDF5data(folder+filename+extension)
xrange, yrange, startframe, endframe, quicklook = \
    dfsm.movie_properties(data, movienumber)
mean_img = np.mean(data[movienumber,1][:,:,
                   startframe:endframe:int((endframe-startframe)/10)], axis=2)

# Synchronize DAC and camera:
moviesignaloffset = dfsm.movie_signal_offset(quicklook, DAC_led)
mof, sof, msmax = \
    dfsm.create_offset_parameters(moviesignaloffset, quicklook, DAC_led)


#%%
# Look at "framenumber" frame of "movienumber" movie and show all data tracks normalized:
framenumber = 30

plt.figure(figsize=(10,6))
plt.imshow(data[movienumber,1][:,:,framenumber].T)
plt.title("Frame")
plt.show()

def scale(data):
    npdata = np.array(data) ; ma=np.max(npdata) ; mi=np.min(npdata)
    return (npdata-mi)/np.abs(ma-mi)

plt.figure(figsize=(10,6))
plt.plot(scale(quicklook))
plt.plot(scale(DAC_v))
plt.plot(scale(DAC_c))
plt.plot(scale(DAC_led))
plt.title("Daq stuff, scaled")
plt.show()

print("xrange", xrange, "\nyrange", xrange, "\nframes", endframe-startframe)


#%%
# Guess particle and get its PDOC in said range

x_guess = 10
y_guess = 10

# Get location:
x_found, y_found = \
    dfsm.find_max_I_around_guess(mean_img, x_guess, y_guess, areasize=15)
x1, x2, y1, y2 = dfsm.getROIaroundXYinFigure(x_found, y_found, mean_img, xsize=1, ysize=1)
x_range, y_range = (x1, x2), (y1, y2)


results = []
for signal in signals:
    equilibriate_skip = signal["period"]*4
    particle = np.sum(np.sum(data[movienumber,1][x_range[0]:x_range[1], y_range[0]:y_range[1],int(signal["range"][0]+equilibriate_skip+mof):int(signal["range"][1]-sof)] - 96, axis=0), axis=0)
    #particle = (np.roll(particle,-1)+particle)/2  # This is sometimes necessary
    adjusted_signal = DAC_v[int(signal["range"][0]+equilibriate_skip+sof):int(signal["range"][1]-mof)]
    adjusted_current = DAC_c[int(signal["range"][0]+equilibriate_skip+sof):int(signal["range"][1]-mof)]
    driftcorrected_particle = dfsm.correct_particle_drift(particle, polyorfft='poly', polyorder=12, fftfilter=16)
    fig = plt.figure(figsize=(12,6))
    plt.plot(particle, color="xkcd:orange", label="signal before correction")
    plt.plot(driftcorrected_particle, color='xkcd:green', label="signal after correction")
    outlier_rate = 0.2  # 0.2 should only filter truely bad parts.  
    i, v, c = dfsm.cycleaverage_particle_and_DAC(driftcorrected_particle, 
        adjusted_signal, adjusted_current, signal["period"], signal["phase"], outlier_rate)
    plt.title("Drift correction and accepted cycles", size=16)
    plt.xlabel("frames since signal start", size=14)
    plt.ylabel("intensity (counts)", size=14)
    plt.legend() ; plt.show()
    roll_amount = - np.argmax(v) # + int(1/4*signal["period"])
    i = np.roll(i, roll_amount) ; v = np.roll(v, roll_amount) 
    c = np.roll(c, roll_amount) ; t_arr = Tframe*np.arange(len(i))
    results.append({"cycle_potential": v, "cycle_intensity": i, "cycle_current": c,
                    "potential": adjusted_signal, "intensity": driftcorrected_particle, 
                    "x_range": x_range, "y_range":y_range})
    # Autoplot stuff:
    fig = plt.figure(figsize=(10,4)) ; ax1 = fig.add_subplot(1,1,1)
    color = 'tab:red' ; ax1.set_xlabel('time ($s$)', size=16)
    ax1.set_ylabel('$\mathrm{\Phi}_{\mathrm{cell}}$ ($V$)', size=16, color=color) 
    ax1.plot(t_arr, v, '.', color=color) ; ax1.tick_params('y', colors=color)
    ax2 = ax1.twinx() ; color = 'xkcd:royal blue'
    ax2.set_ylabel('I/I$_{avr}$ ', size=16, color=color)
    ax2.plot(t_arr, i/np.mean(i), '.', color=color)
    ax2.tick_params('y', colors=color) ; plt.show()
    separateupdown = [np.argmax(v), np.argmin(v)] ; half = int(len(i)/2 + 1)
    fig = plt.figure(figsize=(10,6)) ; ax1 = fig.add_subplot(1,1,1)
    # Uncomment to show every single point:
    #ax1.plot(adjusted_signal, driftcorrected_particle/np.mean(i), '.', markersize=2, color='xkcd:grey', label="All cycles") 
    ax1.set_xlabel('$\Phi_{cell}$ (V)', size=18) ; ax1.set_ylabel('$I/I_{avr}$', size=18)
    ax1.plot(np.roll(v, separateupdown[0])[:half], np.roll(i, separateupdown[0])[:half]/np.mean(i), 
             '.-b', markersize=15, label="decreasing potential")
    ax1.plot(np.roll(v, separateupdown[1])[:half], np.roll(i, separateupdown[1])[:half]/np.mean(i), 
             '.-r', markersize=15, label="increasing potential")
    ax1.tick_params(labelsize=14) ; plt.legend() ; plt.show()
    print("For offset:", signal["offset"])

#%%
# Room for more code:

# For example, for result '0':
plt.figure()
plt.plot(results[0]['cycle_potential'], results[0]['cycle_intensity']/np.mean(results[0]['cycle_intensity']))
plt.xlabel('potential')
plt.ylabel('intensity')
plt.show()

plt.figure()
plt.plot(results[0]['cycle_potential'], results[0]['cycle_current'])
plt.xlabel('potential')
plt.ylabel('current')
plt.show()

























