# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 09:50:16 2019

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import functionsDFSM as dfsm
import os

# Import data:
folder = "C:\\Data\\Kevin\\UUTrack\\2019-04-17"
filename = "BinningTest"
extension = ".hdf5"

dirName = folder+filename
if not os.path.exists(dirName):
    os.mkdir(dirName)

data = dfsm.ImportHDF5data(folder+filename+extension)
data.setkey(0)
print("Wherein there is : ", data.getkeys())
data.resetkeys()

info = """
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

plt.imshow(data[movienumber, 1][:,:,framenumber])  # This works only for movie data.
plt.show()


#%% 
# NOTE: this takes some time
# Give movie of interest an easier name, filter empty frames and declare variables for later:
Tsignal = 1/2.  # In seconds
Asignal = 1.  # In volts 
Tframe = 1/200.  # In seconds
FPsignal = Tsignal/Tframe  # This variable is used sometimes, but I think it kinda forces the result.


movie = data[movienumber,1]
xrange, yrange, Nmax = movie.shape
metadata = data[movienumber,0]

for i in range(Nmax):
    if (max(movie[:,0,i]) != 0):
        startframe = (i)
        break

for i in range(Nmax):
    if (max(movie[:,0,Nmax-i-1]) != 0):
        endframe = (Nmax-i-1)
        break
    
movie = data[movienumber,1][:,:,startframe:endframe]
Nperiods = (endframe-startframe)/FPsignal  # This variable is often used

print("startframe: %d \nendframe: %d \nxrange: 0 to %d \nyrange: 0 to %d \nNperiods: %lf (%d frames per signal in %d frames)" %(startframe, endframe, xrange, yrange, Nperiods, FPsignal, endframe-startframe))

#%%
# NOTE: don't change stuff in this block yet. This blocks takes a little bit of time.
# Isolate region of interest. This block declares functions and shows the full picture once.
# Interesting is varI/I, where this is high valued the video shows more interesting stuff

viewframeNumber = 0  # Number of frame shown as example
xmin, xmax = 0, xrange  # Set both x and y range of interesting area (to reduce time producing var plot)
ymin, ymax = 0, yrange
xofinterest, yofinterest = int(xrange/2), int(yrange/2)  # Chose an interesting pixel from the varI/I plot (Shown as red dot in it)

def showROIregion():
    plt.imshow(movie[:,:,viewframeNumber], extent=(0,yrange,xrange,0))
    plt.hlines(xmin,0,yrange, 'r')
    plt.hlines(xmax,0,yrange, 'r')
    plt.vlines(ymin,0,xrange, 'r')
    plt.vlines(ymax,0,xrange, 'r')
    plt.title("Intensity")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.show()

def showROIint():
    plt.imshow(movie[xmin:xmax,ymin:ymax,viewframeNumber], extent=(ymin,ymax,xmax,xmin))
    clb = plt.colorbar()
    clb.set_label('Intensity (arb.u.)')
    plt.title("Intensity")
    plt.scatter(yofinterest, xofinterest, s=2, c='red', marker='o')
    plt.xlabel("y (pixels)")
    plt.ylabel("x (pixels)")
    plt.show()

def showROIvar():
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
    
showROIint()
showROIvar()



#%%
# Now set ROI using xmin to ymax. x and y of interest can be changed a bit in the next part
viewframeNumber = startframe  # Number of frame shown as example
xmin, xmax = 675, 800  # Set both x and y range of interesting area (to reduce time producing var plot)
ymin, ymax = 0, yrange
xofinterest, yofinterest = 771, 77  # Chose an interesting pixel from the varI/I plot (Shown as red dot in it)

showROIregion()
showROIint()


spotintime = movie[xofinterest, yofinterest, :]  # Rename the data of the interesting spot/pixel.

#%% 
# Plot var plot after getting rid of the big drift by using FFT:
filteruntil = 10  # FFT filter values
filterfrom = 1000
xofinterest, yofinterest = xofinterest, yofinterest

def filterMovie(movie, filteruntil, filterfrom):
    # Applies FFT filter to a movie, shrinks ROI around pixel of interest if needed
    xminf = max([xmin, xofinterest-10])
    xmaxf = min([xmax, xofinterest+10])
    yminf = max([ymin, yofinterest-10])
    ymaxf = min([ymax, yofinterest+10])
    partOfMovieForVarPlot = movie[xminf:xmaxf,yminf:ymaxf,:]
    modPartOfMovieForVarPlot = np.zeros_like(partOfMovieForVarPlot)
    if len(modPartOfMovieForVarPlot[0,0,:])%2==1:  # Fix a thing with uneven frame numbers
        modPartOfMovieForVarPlot=modPartOfMovieForVarPlot[:,:,:-1]
    for i in range(len(partOfMovieForVarPlot[:,0,0])):
        for j in range(len(partOfMovieForVarPlot[0,:,0])):
            modPartOfMovieForVarPlot[i,j,:] = dfsm.pixelFFTFilter(partOfMovieForVarPlot[i,j,:], filteruntil, filterfrom)
    return partOfMovieForVarPlot, modPartOfMovieForVarPlot


partOfMovieForVarPlot, modPartOfMovieForVarPlot = filterMovie(movie, filteruntil, filterfrom)

varPlot = (np.var(modPartOfMovieForVarPlot, axis=2)+np.mean(partOfMovieForVarPlot, axis=2))/(np.mean(partOfMovieForVarPlot, axis=2)-96)
plt.imshow(varPlot, origin='upper', extent=(max([ymin, yofinterest-10]),min([ymax, yofinterest+10]),min([xmax, xofinterest+10]),max([xmin, xofinterest-10])))
clb = plt.colorbar()
clb.set_label('(arb.u.)')
plt.title("VarI/I after FFT filtering big drifts")
plt.xlabel("y")
plt.ylabel("x")
plt.scatter(yofinterest, xofinterest, s=2, c='red', marker='o')
plt.savefig(folder+filename+"/VarIoverI-ROI-filtered_"+filename)
plt.show()


#%%
# NOTE: edit lower part of block to change what is shown.
# Filter the FFT of the spot of interest intensity, then get the correlation
filteruntil = 30  # FFT filter values
filterfrom = 1000
uglyLoweredgeAfterFFTFilter = 200  # Spot after FFT filter has ugly edges usually, filter with this before correlation calculation
uglyUpperedgeAfterFFTFilter = -200
timeaxis = 'f' # Set time axis, 's' for seconds, 'f' for frames

def spotShowFFT(): # Make true to show fourier transform
    fspotintime = np.fft.rfft(spotintime)
    plt.plot(np.arange(filteruntil,filterfrom), np.abs(fspotintime[filteruntil:filterfrom]))
    plt.yscale('log')
    plt.title("FFT of intensity at spot after filtering")
    plt.show()

def spotBeforeFFTF(): # Make true to show spot in time before FFT filter 
    spotintime2 = spotintime
    if (timeaxis == 's'): 
        plt.plot(np.arange(0, Tframe*len(spotintime2), Tframe), spotintime2)
    else: 
        plt.plot(spotintime2)
    plt.title("Spot of interest before filtering some Fourier components")
    if (timeaxis == 's'): 
        plt.xlabel("Time (s)")
    else: 
        plt.xlabel("Time (frames)")
    plt.ylabel("Intensity (arb. u.)")
    plt.savefig(folder+filename+"/Intensity_SOI-preFFT_"+filename)
    plt.show()

spotInTimeFiltered = dfsm.pixelFFTFilter(spotintime, filteruntil, filterfrom)

def spotAfterFFTF():  # Make true to show spot in time after FFT filter
    if (timeaxis == 's'): 
        plt.plot(np.arange(0, Tframe*len(spotintime), Tframe), spotInTimeFiltered)
    else: 
        plt.plot(spotInTimeFiltered)
    plt.title("Spot of interest after filtering some Fourier components")
    if (timeaxis == 's'): 
        plt.xlabel("Time (s)")
    else: 
        plt.xlabel("Time (frames)")
    plt.ylabel("Intensity (arb. u.)")
    plt.savefig(folder+filename+"/Intensity_SOI-postFFT_"+filename)
    plt.show()

correlationAfterFramesFFT = dfsm.timeDependentCorrelation(spotInTimeFiltered[uglyLoweredgeAfterFFTFilter:uglyUpperedgeAfterFFTFilter])

def plotCorrelation(): # Get time dependent correlation after cutting of the ugly edges:
    if (timeaxis == 's'): 
        plt.plot(np.arange(0, Tframe*int(9*FPsignal), Tframe), correlationAfterFramesFFT[0:int(9*FPsignal)])
    else: 
        plt.plot(correlationAfterFramesFFT[0:int(9*FPsignal)])
    plt.title("Selfcorrelation over timeinterval")
    if (timeaxis == 's'): 
        plt.xlabel("Timeinterval (s)")
    else: 
        plt.xlabel("Timeinterval (frames)")
    plt.ylabel("Correlation")
    plt.savefig(folder+filename+"/Correlation_SOI-postFFT_"+filename)
    plt.show()

# Comment/uncomment to show or not:
#spotShowFFT()
spotBeforeFFTF()
spotAfterFFTF()
plotCorrelation()

#%%
# Find period from selfcorrelation over timeinterval
approximateFirstDip = int(FPsignal/2)
numberOfPeaksAveraged = int(Nperiods-1)

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
print("Expecting periodicity of: " + str(FPsignal))
print("Found a periodicity of: " + str(foundPeriod))
print("From peaks in correlation at: ")
print(periodmax)




if foundPeriod > FPsignal-5 and foundPeriod < FPsignal+5:
    print("Signal found in video")
else:
    print("Signal not found in video, crashing on purpose:")
    raise Exception('Fail on purpose to stop file execution, because the next code will not be helpfull.')













# NOTE: the next parts are still work in progress and only work if a periodicity was found that is expected for the used potential






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
plt.savefig(folder+filename+"/AveragedPeriod_SOI-postFFT_"+filename)
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
# NOTE: I hope I am doing this wrong, since I don't find any correlation.
# Find correlation distance around interesting pixel
distanceOfPixelsToTry = 10

pixelDistanceArr = np.arange(-distanceOfPixelsToTry, distanceOfPixelsToTry)
correlationDistanceArr = np.array([ np.correlate(spotintime, movie[xofinterest+i, yofinterest, :]) 
                           for i in pixelDistanceArr])

plt.plot(pixelDistanceArr, correlationDistanceArr[:,1])
plt.title("Correlation over distance")
plt.xlabel("Distance (pixels)")
plt.ylabel("Correlation")
plt.show()

























