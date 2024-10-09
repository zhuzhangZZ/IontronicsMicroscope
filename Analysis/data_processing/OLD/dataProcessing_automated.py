# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 09:50:16 2019

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import functionsDFSM as dfsm
import os
from scipy.optimize import curve_fit


def process(folder, filename, movienumber, Tframe, Tsignal, Asignal, savedirName = 0):
#%%
    if __name__ == '__main__':
        # Import data:
        folder = "/media/kevin/My Passport/2019-02-18-measurements/"
        filename = "NaI-Ptw3"
        extension = ".hdf5"
        movienumber = 3
        Tframe = 1/200.  # In seconds the time per frame 
        
        averageforspot = True  # Take average of 9 pixels for the spot of interest or not
        hertz = 1
        Tsignal = 1./hertz  # In seconds, used a bit
        Asignal = 1.  # In volts, never used
        savedirName = 0
        
    
    averageforspot = True  # Take average of 9 pixels for the spot of interest or not
    extension = ".hdf5"
    if savedirName == 0:
        dirName = folder+filename+"_m"+str(movienumber)
        if not os.path.exists(dirName):
            os.makedirs(dirName)
    else:
        dirName = savedirName
        if not os.path.exists(dirName):
            os.makedirs(dirName)
    
    data = dfsm.ImportHDF5data(folder+filename+extension)
    data.setkey(0)
    print("Wherein there is : ", data.getkeys())
    data.resetkeys()
    

    
    def customSave(figurename, plotted):
        # Needs to be called when the figure is in "memory" needs a logical name of the figure and the plotted np data.
        # Use to save the figure currently in use and the raw np data used to make it, puts it in the correct folder.
        previewfile = dirName+"/"+figurename+"_"+filename+'_rawImage.npy'
        np.save(previewfile, plotted)
        plt.savefig(dirName+"/"+figurename+"_"+filename)
        
    
    #%% 
    # Check start and end of data and plot intensity of middle line over time:
    
    FPsignal = Tsignal/Tframe  
    xrange, yrange, Nmax = data[movienumber-1,1].shape
    
    middleline = np.mean(data[movienumber-1,1][:,int(yrange/2),:], axis=0)
    nf = np.size(middleline[middleline>0])
    plt.plot(middleline[:nf])
    plt.title("Average of the center line")
    plt.xlabel("Frame number")
    plt.ylabel("Intensity (arb.u.)")
    plt.savefig(dirName+"/Average-Center-Line_"+filename)
    plt.show()
    
    startframe = 0
    endframe = nf
    
    movie = np.array(data[movienumber-1,1][:,:,startframe:endframe])
    Nperiods = (endframe-startframe)/FPsignal  # This variable is used for knowing how many peaks to find for the self correlation
    
    print("startframe: %d \nendframe: %d \nxrange: 0 to %d \nyrange: 0 to %d \nNperiods: %lf (%d frames per signal in %d frames)" %(startframe, endframe, xrange, yrange, Nperiods, FPsignal, endframe-startframe))
    
    #%%
    # Time average of movie:
    
    meanfigure = np.mean(movie[:,:,:], axis=2)
    
    plt.imshow(meanfigure, origin='lower')
    clb = plt.colorbar()
    clb.set_label('(arb.u.)')
    plt.title("Averaged Intensity")
    plt.xlabel("y")
    plt.ylabel("x")
    
    figurename = "Time-Averaged-Picture"
    plotted = meanfigure
    customSave(figurename, plotted)
        
    plt.show()
    
    
    #%%
    # Start looking for spot of interest by making a plot of the variance of the picture: 
    
    safelength = 4000
    
    movieTimeLength = endframe-startframe
    safeendframe, safestartframe = endframe, startframe
    if movieTimeLength > safelength:
        safeendframe = endframe - int((movieTimeLength-safelength)/2)
        safestartframe = startframe + int((movieTimeLength-safelength)/2)
        
    partOfMovieForVarPlot = np.array(movie[:,:,safestartframe:safeendframe]).copy()
    
    mean_img = np.mean(partOfMovieForVarPlot, axis=2)
    dark = np.min(mean_img)
    mean_img = mean_img - dark + 1
    
    varPlot = np.var(partOfMovieForVarPlot, axis=2)/mean_img
    
    #%%
    # Find spot of interest, which is the spot with maximum variance: (with some limitations)
    
    varPlotMOD = varPlot.copy()
    varPlotMOD[varPlotMOD>np.mean(varPlot)*4] = 0  # Filter very high values because they are probably irregularities.
    
    max_var_pos_x, max_var_pos_y = np.unravel_index(np.argmax(varPlotMOD), varPlot.shape)
    
    # Make plot of area:
    # First decide how big an area around it should be plotted:
    xmin, xmax = 0, xrange
    ymin, ymax = 0, yrange
    maxsizearoundpoint = 50
    xminf = max([xmin, max_var_pos_x-maxsizearoundpoint])
    xmaxf = min([xmax, max_var_pos_x+maxsizearoundpoint])
    yminf = max([ymin, max_var_pos_y-maxsizearoundpoint])
    ymaxf = min([ymax, max_var_pos_y+maxsizearoundpoint])
    
    if False:  # Make True to see the spot in the whole image:
        plt.imshow(varPlot[0:xrange,0:yrange], extent=(0,yrange,0,xrange), clim=(0,500), origin='lower')
        clb = plt.colorbar()
        clb.set_label('(arb.u.)')
        plt.title("VarI/I around maximum spot")
        plt.xlabel("y")
        plt.ylabel("x")
        plt.scatter(max_var_pos_y, max_var_pos_x, s=2, c='red', marker='o')
        plt.show()
    
    # Plot and save:
    plt.imshow(varPlot[xminf:xmaxf,yminf:ymaxf], extent=(yminf,ymaxf,xminf,xmaxf), origin='lower')
    clb = plt.colorbar()
    clb.set_label('(arb.u.)')
    plt.title("VarI/I around maximum spot")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.scatter(max_var_pos_y, max_var_pos_x, s=2, c='red', marker='o')
    plt.savefig(dirName+"/Variance-Around-Chosen-Maximum-Spot_"+filename)
    plt.show()
    
    #%%
    # Find the spot with minimum variance:
    
    min_var_pos_x, min_var_pos_y = np.unravel_index(np.argmin(varPlot), varPlot.shape)
    
    # Make plot of area:
    xmin, xmax = 0, xrange
    ymin, ymax = 0, yrange
    xminf = max([xmin, min_var_pos_x-maxsizearoundpoint])
    xmaxf = min([xmax, min_var_pos_x+maxsizearoundpoint])
    yminf = max([ymin, min_var_pos_y-maxsizearoundpoint])
    ymaxf = min([ymax, min_var_pos_y+maxsizearoundpoint])
    plt.imshow(varPlot[xminf:xmaxf,yminf:ymaxf], extent=(yminf,ymaxf,xminf,xmaxf), origin='lower')
    clb = plt.colorbar()
    clb.set_label('(arb.u.)')
    plt.title("VarI/I around minimum spot")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.scatter(min_var_pos_y, min_var_pos_x, s=2, c='red', marker='o')
    plt.show()
    
    
    #%%
    # Setup for looking at harmonic components and more and potentially make sure the SOI is interesting
    
    maxFFTc = 100 # fourier index above which is irrelevant for the analysis
    dcbase = 10    # fourier index below which counts as drift
    
    xofinterest, yofinterest = max_var_pos_x, max_var_pos_y
    
    maxsizearoundpoint = 20
    xminf = max([xmin, max_var_pos_x-maxsizearoundpoint])
    xmaxf = min([xmax, max_var_pos_x+maxsizearoundpoint])
    yminf = max([ymin, max_var_pos_y-maxsizearoundpoint])
    ymaxf = min([ymax, max_var_pos_y+maxsizearoundpoint])
    partOfMovieForFFTVarPlot = partOfMovieForVarPlot[xminf:xmaxf,yminf:ymaxf,safestartframe:safeendframe]
    
    if False: # This is not necessary and takes long, so lets not do it
        # Automatically find spot of interest in range around previously found spot,
        # but now after fourier filtering the large and small components.
        ROI_after_FFT = np.zeros_like(partOfMovieForFFTVarPlot)
        if len(ROI_after_FFT[0,0,:])%2==1:  # Fix a thing with uneven frame numbers and FFT
            ROI_after_FFT=ROI_after_FFT[:,:,:-1]
        for i in range(len(partOfMovieForFFTVarPlot[:,0,0])):
            for j in range(len(partOfMovieForFFTVarPlot[0,:,0])):
                ROI_after_FFT[i,j,:] = dfsm.pixelFFTFilter(partOfMovieForFFTVarPlot[i,j,:], dcbase, maxFFTc)
                
        mean_img = np.mean(partOfMovieForFFTVarPlot, axis=2)
        dark = np.min(mean_img)
        mean_img = mean_img - dark + 1
        
        varPlotFFT = np.var(ROI_after_FFT, axis=2)/mean_img
    
    
    #%%
    # Looking at the harmonic component and more:
    
    if averageforspot:
        SOI = movie[max_var_pos_x-1:max_var_pos_x+1, max_var_pos_y-1:max_var_pos_y+1, :]
        SOI = np.mean(np.mean(SOI, axis=0), axis=0)
    else:
        SOI = movie[max_var_pos_x, max_var_pos_y, :]
        
    FFTspec = np.fft.rfft(SOI)
    
    
    # Find maximum valued frequency
    mfreq = dcbase + np.argmax(abs(FFTspec[dcbase:maxFFTc]))
    #shfreq = mfreq + dcbase + np.argmax(abs(fspec[mfreq + dcbase:]))
    #thirdfreq = shfreq + dcbase + np.argmax(abs(fspec[shfreq + dcbase:]))
    print("Fourier components:", mfreq)
    
    plt.plot(np.arange(dcbase, maxFFTc), np.log10(abs(FFTspec[dcbase:maxFFTc]))) # Give correct "x"axis
    plt.title("Fourier components of interest")
    plt.xlabel("Fourier component")
    plt.ylabel("Amplitude?")
    plt.show()
    
    
    ffpwidth = 3 #estimated half width of the Fourier peak in the power spectrum
    frames_fft = np.fft.rfft(partOfMovieForFFTVarPlot, axis=2)/len(partOfMovieForFFTVarPlot[0,0,:])
    first_harmonic = np.sum(np.abs(frames_fft[:, :, mfreq-ffpwidth:mfreq+ffpwidth+1]), axis=2)
    
    plt.imshow(first_harmonic, cmap='Greys_r', extent=(yminf,ymaxf,xminf,xmaxf), origin='lower')
    plt.title("First harmonic")
    plt.xlabel("x")
    plt.ylabel("y")
    
    figurename = "first_harmonic"
    plotted = first_harmonic
    customSave(figurename, plotted)
    plt.show()
    
    
    phase = np.angle(frames_fft[:,:, mfreq])/np.pi
    
    plt.imshow(phase, cmap='hsv', extent=(yminf,ymaxf,xminf,xmaxf), origin='lower')
    plt.title("Phase")
    plt.xlabel("x")
    plt.ylabel("y")
    
    figurename = "phase"
    plotted = phase
    customSave(figurename, plotted)
    plt.show()
    
    FFTspecFiltered = FFTspec.copy()
    FFTspecFiltered[:dcbase] = 0
    FFTspecFiltered[maxFFTc:] = 0
    specFiltered = np.fft.irfft(FFTspecFiltered)
    specFiltered = specFiltered[100:-100]  # Trow away the sides of the data because the FFT filter makes them ugly
    
    correlationAfterFramesFFT = dfsm.timeDependentCorrelation(specFiltered)
    
    plt.plot(specFiltered)
    plt.title("Spot of interest after filtering some Fourier components")
    plt.xlabel("Time (frames)")
    plt.ylabel("Intensity (arb. u.)")
    
    figurename = "Intensity_SOI-postFFT"
    plotted = specFiltered
    customSave(figurename, plotted)
    plt.show()
    
    plt.plot(correlationAfterFramesFFT)
    plt.title("Self correlation of spot of interest after\n filtering some Fourier components")
    plt.xlabel("Time (frames)")
    plt.ylabel("Correlation")
    figurename = "Self-Correlation-SOI-postFFT"
    plotted = correlationAfterFramesFFT
    customSave(figurename, plotted)
    plt.show()
    
    
    
    #%%
    
    approximateFirstDip = 0
    for i in range(len(correlationAfterFramesFFT)):
        if correlationAfterFramesFFT[i+2]> correlationAfterFramesFFT[i]:
            approximateFirstDip = i
            break
    numberOfPeaksAveraged = int(Nperiods-1)
    
    def findSignalPeriodFromSelfcorrelation(selfcorrel, lowcut, npeaks):
        maxValAtArr = np.array([0]*(npeaks+1))
        actualpeaks = npeaks
        for i in range(npeaks-1):
            nextstart = maxValAtArr[i]+lowcut
            nextend = nextstart+lowcut*2
            if nextend>len(selfcorrel) or nextstart>len(selfcorrel):
                actualpeaks = i+1
                break
            maxValAtArr[i+1] = nextstart + np.argmax(selfcorrel[nextstart:nextend])
        maxValAtArr = maxValAtArr[:actualpeaks]
        speriod = np.mean(np.diff(maxValAtArr))
        eachperiod = maxValAtArr
        return eachperiod, speriod
        
        
    periodmax, foundPeriod = findSignalPeriodFromSelfcorrelation(correlationAfterFramesFFT, approximateFirstDip, numberOfPeaksAveraged)
    print("Expecting periodicity of: " + str(FPsignal))
    print("Found a periodicity of: " + str(foundPeriod))
    print("From peaks in correlation at: ")
    print(periodmax)
    
    
    plt.plot(correlationAfterFramesFFT)
    plt.title("Self correlation of spot of interest after filtering\n some Fourier components, found period: %.2lf" %(foundPeriod))
    plt.xlabel("Time (frames)")
    plt.ylabel("Correlation")
    
    for i in periodmax:
        plt.plot([i,i], [np.min(correlationAfterFramesFFT),np.max(correlationAfterFramesFFT)], color='r', linestyle='-', linewidth=1)
    
    figurename = "Self-Correlation-SOI-postFFT-autoPeriodMarked"
    plotted = periodmax
    customSave(figurename, plotted)
    plt.show()
    
    
    
    #%%
    # Get how the average period looks like for the spot of interest after FFT filtering
    
    usableFilteredSignal = specFiltered
    n_periods, n_overlap = 1, 0
    
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
    
    foundAveragePeriod = averagePeriod(usableFilteredSignal, FPsignal, n_periods, n_overlap)
    
    plt.plot(foundAveragePeriod)
    plt.title("Averaged period")
    plt.xlabel("Time (frames)")
    plt.ylabel("Signal (arb.u.)")
    plt.savefig(dirName+"/AveragedPeriod_SOI-postFFT_"+filename)
    plt.show()
    
    
    
    #%%
    # Fit a sine to the data:
    
    # Glue one set of data in front and back make it better
    fitoverN = 3
    fitData = np.ravel(np.array([foundAveragePeriod]*fitoverN))
    
    def fitsin(x, a, b, c):
        return a * np.sin(-b*(x - c))
    xx = np.array(range(len(fitData)))
    
    # Set value bounds, might be a bit too forcefull right now:
    amin = 0.5 * max(foundAveragePeriod)
    amax = 1.5 * max(foundAveragePeriod)
    bmin = 0.5 * 2*np.pi/len(foundAveragePeriod)*n_periods
    bmax = 1.5 * 2*np.pi/len(foundAveragePeriod)*n_periods
    cmin = -.5 * len(foundAveragePeriod)
    cmax = 1.5 * len(foundAveragePeriod)
    
    popt, pcov = curve_fit(fitsin, xx, fitData, 
                           p0 = [max(foundAveragePeriod), 2*np.pi/len(foundAveragePeriod)*n_periods, np.argmax(foundAveragePeriod)],
                           bounds=([amin, bmin, cmin], [amax, bmax, cmax]))
    
    
    plt.plot(xx[:int(len(xx)/fitoverN)], fitData[int(len(xx)/fitoverN):int(len(xx)/fitoverN*(fitoverN-1))], 'b.')
    plt.plot(xx[:int(len(xx)/fitoverN)], fitsin(xx[int(len(xx)/fitoverN):int(len(xx)/fitoverN*(fitoverN-1))], *popt), 'r-',
             label='fitfunction: %1.2f sin( %1.2f (x - %1.2f))' % tuple(popt))
    plt.xlabel('Time (frames)')
    plt.ylabel('Signal (arb.u.)')
    plt.legend()
    figurename = "FoundFit"
    customSave(figurename, foundAveragePeriod)
    plt.show()
    print("Optimal values (resp. [a b c]):")
    print(popt)
    print("Covariation matrix:")
    print(pcov)
    
    #%%
    # Save all interesting data:
    
    metadataishthing = np.array([("filename", filename), 
                                 ("movienumber", movienumber), 
                                 ("camera_fps", Tsignal), 
                                 ("frames_per_signal", FPsignal), 
                                 ("signal_amplitude_V", Asignal), 
                                 ("signal_period_seconds", Tsignal), 
                                 ("n_frames", nf), 
                                 ("mfreq", mfreq), 
                                 ("found_amplitude", popt[0]) ], dtype='object, double')
    
    metadataishthingfile = dirName+"/"+data+"_"+filename+'.npy'
    np.save(metadataishthingfile, metadataishthing)
    
    #%%
    # Another way to summerize the data is Sum[I(t).sin(wt)] where w is the 
    # modulation frequency in 1/frames and the sum is taken over all images.
    
    def getSumIsinW(partOfMovieForFFTVarPlot, mfreq):
        sumitsinwt = np.zeros_like(partOfMovieForFFTVarPlot[:,:,0])
        for i in range(len(partOfMovieForFFTVarPlot[:,0,0])):
            for j in range(len(partOfMovieForFFTVarPlot[0,:,0])):
                sumitsinwt[i,j] = sum([partOfMovieForFFTVarPlot[i,j,t]*np.sin(1/mfreq*t) for t in range(len(partOfMovieForFFTVarPlot[0,0,:]))])
        return sumitsinwt
    
    sumitsinwt = getSumIsinW(partOfMovieForFFTVarPlot, mfreq)
    plt.imshow(sumitsinwt, extent=(yminf,ymaxf,xminf,xmaxf), origin='lower')
    plt.title("Sum[I(t).sin(wt)] dt")
    plt.xlabel("x")
    plt.ylabel("y")
    figurename = "Sum-It-sinwt-dt"
    customSave(figurename, sumitsinwt)
    plt.show()

#%%

if __name__ == '__main__':
    # Import data:
    folder = "/media/kevin/My Passport/2019-02-18-measurements/"
    filename = "NaI-Ptw3"
    extension = ".hdf5"
    movienumber = 3
    Tframe = 1/200.  # In seconds the time per frame 
    
    averageforspot = True  # Take average of 9 pixels for the spot of interest or not
    hertz = 1
    Tsignal = 1./hertz  # In seconds, used a bit
    Asignal = 1.  # In volts, never used
    
    process(folder, filename, movienumber, Tframe, Tsignal, Asignal)



