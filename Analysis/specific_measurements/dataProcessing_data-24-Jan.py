# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 09:50:16 2019

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import functionsDFSM as dfsm  # Sanli: May I suggest you make a folder KevinFunctions and put all files that you write or find useful for others in that folder

# Import data:
data = dfsm.ImportHDF5data("/home/kevin/Documents/PDSM_data/2019-01-24_different_salts/Video_NaI_2Hz_200fps.hdf5")
data.setkey(0)
print("Wherein there is : ", data.getkeys())
data.resetkeys()

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
framenumber = 0
movienumber = 0

plt.imshow(data[movienumber,1][:,:,framenumber])
plt.show()


#%%
# Give movie of interest an easier name and declare variables of interest:
Tsignal = 1/2.
Tframe = 1/200. 
FPsignal = Tsignal/Tframe  # This variable is used sometimes, but I think it kinda forces the result.

movie = data[movienumber,1]  #Sanli: what is the second index?
xrange, yrange, Nmax = movie.shape
metadata = data[movienumber,0]


#%%
# Show a few frames within the given framenumber range, to find the interesting timeframe: (there is often some crap at the end)
startframe = 0
endframe = 3000

def show10inRangeOfMovie(startf, endf, movie):
    if (endf-startf < 10):
        print("Less then 10 frames in range.")
        for i in range(endf-startf+1):
            plt.imshow(movie[:,:,i], extent=(0,yrange,0,xrange))
            plt.title("Frame %d " %i)
            plt.show()
    else:
        for i in np.linspace(startf, endf-1, num=10):
            iint = int(i)
            plt.imshow(movie[:,:,iint], extent=(0,yrange,0,xrange))
            plt.title("Frame %d " %iint)
            plt.show()
            print(" ")

show10inRangeOfMovie(startframe, endframe, movie)
Nperiods = (endframe-startframe)/FPsignal  # This variable is often used


#%%
# Isolate region of interest.
# Interesting is varI/I, where this is high valued the video shows more interesting stuff
viewframeNumber = 0  # Number of frame shown as example
xmin, xmax = 1150, 1275  # Set both x and y range of interesting area (to reduce time producing var plot)
ymin, ymax = 5, 140
xofinterest, yofinterest = 1185, 40  # Chose an interesting pixel from the varI/I plot (Shown as red dot in it)

plt.imshow(movie[:,:,viewframeNumber], extent=(0,yrange,xrange,0))
plt.hlines(xmin,0,yrange, 'r')
plt.hlines(xmax,0,yrange, 'r')
plt.vlines(ymin,0,xrange, 'r')
plt.vlines(ymax,0,xrange, 'r')
plt.title("Intensity")
plt.xlabel("y")
plt.ylabel("x")
plt.show()

plt.imshow(movie[xmin:xmax,ymin:ymax,viewframeNumber], extent=(ymin,ymax,xmax,xmin))
clb = plt.colorbar()
clb.set_label('Intensity (arb.u.)')
plt.title("Intensity")
plt.xlabel("y")
plt.ylabel("x")
plt.show()

# Plot varI/I for the variance in time for 1 signal oscilation:
#partOfMovieForVarPlot = movie[xmin:xmax,ymin:ymax,viewframeNumber:int(viewframeNumber+2*FPsignal)]
# Or the whole movie:
partOfMovieForVarPlot = movie[xmin:xmax,ymin:ymax,startframe:endframe]
varPlot = np.var(partOfMovieForVarPlot, axis=2)/(np.mean(partOfMovieForVarPlot, axis=2)-96)
plt.imshow(varPlot, extent=(ymin,ymax,xmax,xmin))
clb = plt.colorbar()
clb.set_label('(arb.u.)')
plt.title("VarI/I")
plt.xlabel("y")
plt.ylabel("x")
plt.scatter(yofinterest, xofinterest, s=2, c='red', marker='o')
plt.show()

spotintime = movie[xofinterest, yofinterest, startframe:endframe]  # Rename the data of the interesting spot/pixel.


#%% 
# Plot var plot after getting rid of the big drift by using FFT:
filteruntil = 10  # FFT filter values
filterfrom = 1000
xofinterest, yofinterest = 1200, 115  # Chose an interesting pixel from the varI/I plot (Shown as red dot in it)

partOfMovieForVarPlot = movie[xmin:xmax,ymin:ymax,startframe:endframe]
modPartOfMovieForVarPlot = np.zeros_like(partOfMovieForVarPlot)
for i in range(len(partOfMovieForVarPlot[:,0,0])):
    for j in range(len(partOfMovieForVarPlot[0,:,0])):
        modPartOfMovieForVarPlot[i,j,:] = dfsm.pixelFFTFilter(partOfMovieForVarPlot[i,j,:], filteruntil, filterfrom)
        

varPlot = (np.var(modPartOfMovieForVarPlot, axis=2)+np.mean(partOfMovieForVarPlot, axis=2))/(np.mean(partOfMovieForVarPlot, axis=2)-96)
plt.imshow(varPlot, extent=(ymin,ymax,xmax,xmin))
clb = plt.colorbar()
clb.set_label('(arb.u.)')
plt.title("VarI/I after FFT filtering big drifts")
plt.xlabel("y")
plt.ylabel("x")
plt.scatter(yofinterest, xofinterest, s=2, c='red', marker='o')
plt.show()


#%%
# FFT version of the correlation at the spot of interest 
# Meaning: filter the FFT of the spot intensity, then get the correlation
filteruntil = 30  # FFT filter values
filterfrom = 1000
uglyLoweredgeAfterFFTFilter = 300  # Spot after FFT filter has ugly edges usually, filter with this before correlation calculation
uglyUpperedgeAfterFFTFilter = 2600

if True: # Make true to show fourier transform

    fspotintime = np.fft.rfft(spotintime)
    plt.plot(np.arange(filteruntil,filterfrom), np.abs(fspotintime[filteruntil:filterfrom]))
    plt.yscale('log')
    #plt.xlim(0,filterfrom*1.05)
    plt.ylim(0,1.4*np.max(fspotintime[filteruntil:filterfrom]))
    plt.title("FFT of intensity at spot after filtering")
    plt.show()

if False:  # Make true to show spot in time before FFT filter 
    spotintime2 = spotintime
    plt.plot(np.arange(0, Tframe*len(spotintime2), Tframe), spotintime2)
    plt.title("Spot of interest before filtering some Fourier components")
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity (arb. u.)")
    plt.show()


spotInTimeFiltered = dfsm.pixelFFTFilter(spotintime, filteruntil, filterfrom)

if False:  # Make true to show spot in time after FFT filter
    plt.plot(np.arange(0, Tframe*len(spotintime), Tframe), spotInTimeFiltered)
    plt.title("Spot of interest after filtering some Fourier components")
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity (arb. u.)")
    plt.show()

# Get time dependent correlation after cutting of the ugly edges:
correlationAfterFramesFFT = dfsm.timeDependentCorrelation(
                                  spotInTimeFiltered[uglyLoweredgeAfterFFTFilter:uglyUpperedgeAfterFFTFilter])
plt.plot(correlationAfterFramesFFT[0:int(9*FPsignal)])
plt.title("Selfcorrelation over timeinterval")
plt.xlabel("Timeinterval (frames)")
plt.ylabel("Correlation")
plt.show()

#%%
# Find period from selfcorrelation over timeinterval:
approximateFirstDip = 50
numberOfPeaksAveraged = 10

def findSignalPeriodFromSelfcorrelation(selfcorrel, lowcut, npeaks):
    maxvalatarr = [lowcut + np.argmax(selfcorrel[lowcut:])]
    nextcut = lowcut
    for i in range(npeaks-1):
        nextcut = maxvalatarr[-1]+lowcut
        maxvalatarr.append(nextcut + np.argmax(selfcorrel[nextcut:nextcut+maxvalatarr[-1]]) - maxvalatarr[-1])
    speriod = np.mean(maxvalatarr)
    eachperiod = [np.sum(maxvalatarr[:i]) for i in range(len(maxvalatarr))]
    return eachperiod, speriod
    
    
periodmax, foundPeriod = findSignalPeriodFromSelfcorrelation(correlationAfterFramesFFT, approximateFirstDip, numberOfPeaksAveraged)
print("Found a periodicity of: " + str(foundPeriod))
print(periodmax)


#%%
# Get how the average period looks like for the spot of interest after FFT filtering
usableFilteredSignal = spotInTimeFiltered[uglyLoweredgeAfterFFTFilter:uglyUpperedgeAfterFFTFilter]
n_periods, n_overlap = 2, 0


def averagePeriod(filteredsignal, periodicity, n_periods, n_overlap):
    # Average using n_periods periods and overlap n_overlap each time
    aperiod = np.zeros(int(n_periods*periodicity))
    periodsinsignal = int(len(filteredsignal)/periodicity)
    n_periodsaveraged = int(periodsinsignal/(n_periods-n_overlap))
    for i in range(n_periodsaveraged):
        thisperiodstart = int(periodicity*i*(n_periods-n_overlap))
        thisperiodend = int(thisperiodstart+periodicity*n_periods)
        aperiod += filteredsignal[thisperiodstart:thisperiodend]/periodsinsignal
    return aperiod 
    
foundAveragePeriod = averagePeriod(usableFilteredSignal, foundPeriod, n_periods, n_overlap)

plt.plot(np.arange(0, Tframe*len(foundAveragePeriod), Tframe), foundAveragePeriod)
plt.title("Averaged period")
plt.xlabel("Time (s)")
plt.ylabel("Signal (arb.u.)")
plt.show()

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

for i in range(1):
    plt.plot(usableFilteredSignal[i*int(foundPeriod):i*int(foundPeriod)+int(foundPeriod*n_periods)])

plt.plot(100*triangleWave(2*np.pi*(np.arange(200)/foundPeriod) + np.pi/12))
plt.plot(foundAveragePeriod, 'k+')
plt.title("Averaged period and each period")
plt.xlabel("Timeinterval (frames)")
plt.ylabel("Signal (arb.u.)")
plt.show()
plt.show()


#%%
# Bin over time (to make the graph smoother) and try to fit some curve. JUST TESTING IF IT WORKS

smoothFAP = (foundAveragePeriod[0:-3]+foundAveragePeriod[1:-2]+foundAveragePeriod[2:-1])/3
plt.plot(smoothFAP)
plt.plot(foundAveragePeriod[1:-2])
plt.title("Averaged period")
plt.xlabel("Timeinterval (frames)")
plt.ylabel("Signal (arb.u.)")

from scipy.optimize import curve_fit
def func(x, a, b, c, d):
    return a * np.sin(b * x - c) + d
popt, pcov = curve_fit(func, np.arange(len(smoothFAP)), smoothFAP, bounds=([20., 0.0, -50, -50], [60., 0.1, 50, 50]))
print(popt)
plt.plot(np.arange(len(smoothFAP)), func(np.arange(len(smoothFAP)), *popt), 'r-')

plt.show()


#%%
# Fourier filter and timecorrelation averaged for 9 pixels around the spot of interest
if False:
    filteruntil = 20  # FFT filter values
    filterfrom = 1000
    uglyLoweredgeAfterFFTFilter = 500  # Spot after FFT filter has ugly edges usually, filter with this before correlation calculation
    uglyUpperedgeAfterFFTFilter = 2500
    
    
    spotareaintime = []
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            spotareaintime.append(movie[xofinterest+i, yofinterest+j, startframe:endframe])
    
    # Do FFTFilter on each pixel and show the mean of the result:
    spotAreaFiltered = [dfsm.pixelFFTFilter(i, filteruntil, filterfrom) for i in spotareaintime]
    
    averageSpotAreaFiltered = np.mean(spotAreaFiltered, axis=0)
    plt.plot(averageSpotAreaFiltered)
    plt.title("Spotarea after filtering Fourier components")
    plt.xlabel("Timeinterval (frames)")
    plt.ylabel("Intensity")
    plt.show()
    
    # Do timedependentcorrelation on each pixel and show the mean of the result:
    spotAreaCorrelation = [dfsm.timeDependentCorrelation(i[uglyLoweredgeAfterFFTFilter:uglyUpperedgeAfterFFTFilter], FPsignal) for i in spotAreaFiltered]
    
    averageSpotAreaCorrelation = np.mean(spotAreaCorrelation, axis=0)
    plt.plot(averageSpotAreaCorrelation[0:int(9*FPsignal)])
    plt.title("Selfcorrelation over timeinterval for spotarea")
    plt.xlabel("Timeinterval (frames)")
    plt.ylabel("Correlation")
    plt.show()

#%%
# Fourier filter and timecorrelation for 9 binned pixels around the spot of interest
filteruntil = 20  # FFT filter values
filterfrom = 1000
uglyLoweredgeAfterFFTFilter = 500  # Spot after FFT filter has ugly edges usually, filter with this before correlation calculation
uglyUpperedgeAfterFFTFilter = 2500
showAllUnfiltered = False

spotareaintime = np.zeros_like(movie[xofinterest, yofinterest, startframe:endframe])
for i in [-1,0,1]:
    for j in [-1,0,1]:
        spotareaintime += movie[xofinterest+i, yofinterest+j, startframe:endframe]
        if showAllUnfiltered: plt.plot(movie[xofinterest+i, yofinterest+j, startframe:endframe])
if showAllUnfiltered: 
    plt.show()

    plt.plot(spotareaintime)
    plt.title("Spotarea before filtering Fourier components")
    plt.xlabel("Timeinterval (frames)")
    plt.ylabel("Intensity")
    plt.show()


# Do FFTFilter on each pixel and show the mean of the result:
spotAreaFiltered = dfsm.pixelFFTFilter(spotareaintime, filteruntil, filterfrom)

plt.plot(spotAreaFiltered)
plt.title("Spotarea after filtering Fourier components")
plt.xlabel("Timeinterval (frames)")
plt.ylabel("Intensity")
plt.show()

# Do timedependentcorrelation on each pixel and show the mean of the result:
spotAreaCorrelation = dfsm.timeDependentCorrelation(spotAreaFiltered[uglyLoweredgeAfterFFTFilter:uglyUpperedgeAfterFFTFilter])

plt.plot(spotAreaCorrelation[0:int(9*FPsignal)])
plt.title("Selfcorrelation over timeinterval for spotarea")
plt.xlabel("Timeinterval (frames)")
plt.ylabel("Correlation")
plt.show()


#%%
# Find correlation distance around interesting pixel
distanceOfPixelsToTry = 10

pixelDistanceArr = np.arange(-distanceOfPixelsToTry, distanceOfPixelsToTry)
correlationDistanceArr = [ np.correlate(spotintime, movie[xofinterest+i, yofinterest, startframe:endframe]) 
                           for i in pixelDistanceArr]

plt.plot(pixelDistanceArr, correlationDistanceArr)
plt.title("Correlation over distance")
plt.xlabel("Distance (pixels)")
plt.ylabel("Correlation")
plt.show()


#%%
# FFT of intensity at spot 

fspotintime = np.fft.rfft(spotintime)
plt.plot((np.abs(fspotintime)**2)[0:200])
plt.yscale('log')
plt.title("FFT of intensity")
plt.show()






















