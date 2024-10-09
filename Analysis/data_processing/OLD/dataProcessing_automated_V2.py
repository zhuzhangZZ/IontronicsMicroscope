# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 09:50:16 2019

@author: Kevin Namink <k.w.namink@uu.nl>

Feedback and Comments by Sanli Faez
"""

# COMMENTS ON COMMENTS:
# varPlotMOD[varPlotMOD>np.mean(varPlot)*6] = 0  # Filter very high values because they are probably irregularities. --> 
#SF: Not sure of this judgement. The can well be the points of most interest. Start by looking at the whole image. If you want to 
#K: I expected you to find this and I was more or less aware its a bit ugly.
# I think your solution (varPlot>100) is more scientificly defendable but not much different.
#fig = plt.figure(figsize=(18,5)) #SF: helps getting a more proportionate magnification
#K: This doesn't work for different ROI ratios (square ROI will make this fail).



import numpy as np
import matplotlib.pyplot as plt
import os

# To import our library with functions you might need to put the functionsDFSM.py file in the same folder as this file
import functionsDFSM as dfsm

def process(folder, filename, movienumber, Tframe, Tsignal, Asignal, savedirName = 0):
    #%%
    # Settings
    # Configure settings, make folders, open data files and declare some functions:
    
    # Configure:
    #folder = "/media/kevin/My Passport/2019-02-26-measurements/"
    #filename = "nocells"
    #movienumber = 3
    #Tframe = 1/30.  # In seconds the time per frame 
    #
    #averageforspot = True  # Take average of 9 pixels for the spot of interest or not
    #hertz = 1
    #Tsignal = 1./hertz  # In seconds, used a bit
    #Asignal = 1.  # In volts, never used
    #savedirName = 0  # Switch needed when making file callable
    
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
    for f in ["/overview", "/region_high", "/region_low", "/bright_spot"]:
        if not os.path.exists(dirName+f):
            os.makedirs(dirName+f)
    
    # Open data:
    data = dfsm.ImportHDF5data(folder+filename+extension)
    data.setkey(0)
    print("Wherein there is : ", data.getkeys())
    data.resetkeys()
    
    # Some functions that are easier to declare here:
    def customSave(figurename, plotted):
        # Needs to be called when the figure is in "memory" needs a logical name of the figure and the plotted np data.
        # Use to save the figure currently in use and the raw np data used to make it, puts it in the correct folder.
        previewfile = dirName+"/"+figurename+"_"+filename+'_rawImage.npy'
        np.save(previewfile, plotted)
        plt.savefig(dirName+"/"+figurename+"_"+filename)
        
    def getROIaroundXYinFigure(x, y, figure, maxsizearoundpoint = 75):
        xmin, ymin = 0, 0
        xmax, ymax = figure.shape
        xminf = max([xmin, x-maxsizearoundpoint])
        xmaxf = min([xmax, x+maxsizearoundpoint])
        yminf = max([ymin, y-maxsizearoundpoint])
        ymaxf = min([ymax, y+maxsizearoundpoint])
        return xminf, xmaxf, yminf, ymaxf
    
    
    #%% 
    # Check start and end of data and plot intensity of middle line over time:
    
    FPsignal = Tsignal/Tframe  
    xrange, yrange, Nmax = data[movienumber-1,1].shape
    
    middleline = np.mean(data[movienumber-1,1][:,int(yrange/2),:], axis=0)
    nf = np.argwhere(middleline==0)[0,0]
    plt.plot(middleline[:nf])
    plt.title("Average of the center line")
    plt.xlabel("Frame number")
    plt.ylabel("Intensity (arb.u.)")
    if save_automatically: 
        plt.savefig(dirName+"overview/Average-Center-Line_"+filename)
    plt.show()
    
    startframe = 0
    endframe = nf
    
    movie = np.array(data[movienumber-1,1][:,:,startframe:endframe])
    Nperiods = (endframe-startframe)/FPsignal  # This variable is used for knowing how many peaks to find for the self correlation
    
    print("startframe: %d \nendframe: %d \nxrange: 0 to %d \nyrange: 0 to %d \nNperiods: %lf (%d frames per signal in %d frames)" %(startframe, endframe, xrange, yrange, Nperiods, FPsignal, endframe-startframe))
    
    #%%
    # Time average of movie:
    
    meanfigure = np.mean(movie[:,:,:], axis=2)
    
    #fig = plt.figure(figsize=(18,5)) #SF: helps getting a more proportionate magnification
    plt.imshow(np.log10(np.transpose(meanfigure))) #SF: show transpose to resemble the actual image, use logarithm to see faint features
    clb = plt.colorbar()
    clb.set_label('(arb.u.)')
    plt.title("Logarithm of Averaged Intensity")
    plt.xlabel("x")
    plt.ylabel("y")
    
    if save_automatically: 
        figurename = "overview/Time-Averaged-Picture"
        plotted = meanfigure
        customSave(figurename, plotted)
    
    plt.show()
    
    #%%
    # Start looking for spot of interest by making a plot of the variance of the picture: 
    
    safelength = 1000  # Throw away part of the data if the data is too long to process quickly. 
    
    movieTimeLength = endframe-startframe
    safeendframe, safestartframe = endframe, startframe
    if movieTimeLength > safelength:
        safeendframe = endframe - int((movieTimeLength-safelength)/2)
        safestartframe = startframe + int((movieTimeLength-safelength)/2)
    
    partOfMovieForVarPlot = np.array(movie[:,:,safestartframe:safeendframe]).copy()
    
    mean_img = np.mean(partOfMovieForVarPlot, axis=2)
    dark = np.full(mean_img.shape, 96) # Measured dark field
    dark[dark>mean_img] = mean_img[dark>mean_img] # Set dark field to be lower than usual where the mean image has lower valued pixels, to fix dividing by zero
    mean_img = mean_img - dark + 1
    
    varPlot = np.var(partOfMovieForVarPlot, axis=2)/mean_img
    print("Max and min of Var I/I:", np.max(varPlot), np.min(varPlot))
    
    varPlotMOD = varPlot.copy()
    varPlotMOD[varPlot<1] = 1  # To get rid of saturated pixels that will show sub-shot noise variance
    varPlotMOD[varPlot>100] = 1  # To get rid of exceptionally irregular points
    print("Filtered: " + str((np.sum(varPlot>100)+np.sum(varPlot<1))/varPlot.size) + " %")
    
    #Make plot of area:
    fig = plt.figure(figsize=(18,10))
    
    ax1 = plt.subplot(211)
    im1 = ax1.imshow(np.log10(np.transpose(meanfigure))) #SF: show transpose to resemble the actual image, use logarithm to see faint features
    plt.title("Logarithm of Averaged Intensity")
    plt.ylabel("y")
    clb = plt.colorbar(im1, ax = ax1)
    clb.set_label('(arb.u.)')
    
    ax2 = plt.subplot(212)
    im2 = ax2.imshow(np.log10(np.transpose(varPlotMOD)))
    plt.title("Logarithm of VarI/I")
    plt.xlabel("x")
    plt.ylabel("y")
    clb = plt.colorbar(im2, ax = ax2)
    clb.set_label('(arb.u.)')
    
    if save_automatically: 
        figurename = "overview/Variance"
        plotted = varPlot
        customSave(figurename, plotted)
    plt.show()
    
    #%%
    # Find spot of interest, which is the spot with maximum variance: (with some limitations)
    
    max_var_pos = np.unravel_index(np.argmax(varPlotMOD), varPlot.shape)
    
    #Make plot of area:
    xmin, xmax, ymin, ymax = getROIaroundXYinFigure(max_var_pos[0], max_var_pos[1], varPlot)
    
    fig = plt.figure(figsize=(8,6))
    plt.imshow(np.log10(np.transpose(varPlot[xmin:xmax,ymin:ymax])), extent=(xmin,xmax,ymin,ymax))
    clb = plt.colorbar()
    clb.set_label('log (arb.u.)')
    plt.title("Log VarI/I around maximum spot (var = %.1lf)" % varPlot[max_var_pos[0], max_var_pos[1]])
    plt.xlabel("y")
    plt.ylabel("x")
    if save_automatically: 
        plt.savefig(dirName+"region_high/Variance-Around-Chosen-Maximum-Spot_LOG_"+filename)
    plt.show()
    
    fig = plt.figure(figsize=(8,6))
    plt.imshow((np.transpose(varPlot[xmin:xmax,ymin:ymax])), extent=(xmin,xmax,ymin,ymax))
    clb = plt.colorbar()
    clb.set_label('(arb.u.)')
    plt.title("VarI/I around maximum spot (var = %.1lf)" % varPlot[max_var_pos[0], max_var_pos[1]])
    plt.xlabel("y")
    plt.ylabel("x")
    plt.scatter(max_var_pos[0], max_var_pos[1], s=5, c='red', marker='o')
    if save_automatically: 
        plt.savefig(dirName+"region_high/Variance-Around-Chosen-Maximum-Spot_"+filename)
    plt.show()
    
    
    #%%
    # Find the spot with minimum variance:
    
    min_var_pos = np.unravel_index(np.argmin(varPlot), varPlot.shape)
    
    # Make plot of area:
    xmin, xmax, ymin, ymax = getROIaroundXYinFigure(min_var_pos[0], min_var_pos[1], varPlot)
    
    fig = plt.figure(figsize=(8,6))
    plt.imshow(np.log10(np.transpose(varPlot[xmin:xmax,ymin:ymax])), extent=(xmin,xmax,ymin,ymax))
    clb = plt.colorbar()
    clb.set_label('Log (arb.u.)')
    plt.title("Logarithm VarI/I around minimum spot (var = %.1lf)" % varPlot[min_var_pos[0], min_var_pos[1]])
    plt.xlabel("y")
    plt.ylabel("x")
    if save_automatically: 
        plt.savefig(dirName+"region_low/Variance-Around-Chosen-Minimum-Spot_LOG_"+filename)
    plt.show()
    
    fig = plt.figure(figsize=(8,6))
    plt.imshow((np.transpose(varPlot[xmin:xmax,ymin:ymax])), extent=(xmin,xmax,ymin,ymax))
    clb = plt.colorbar()
    clb.set_label('(arb.u.)')
    plt.title("Logarithm VarI/I around minimum spot (var = %.1lf)" % varPlot[min_var_pos[0], min_var_pos[1]])
    plt.xlabel("y")
    plt.ylabel("x")
    plt.scatter(min_var_pos[0], min_var_pos[1], s=5, c='red', marker='o')
    if save_automatically: 
        plt.savefig(dirName+"region_low/Variance-Around-Chosen-Minimum-Spot_"+filename)
    plt.show()
    
    
    #%%
    # Look at harmonic components using FFT transforms:
    
    x, y = max_var_pos
    
    xmin, xmax, ymin, ymax = getROIaroundXYinFigure(x, y, movie[:,:,0])
    partOfMovieForFFT = np.array(movie[xmin:xmax,ymin:ymax,safestartframe:safeendframe]).copy()
    
    maxFFTc = 100 # fourier index above which is irrelevant for the analysis
    dcbase = 5   # fourier index below which counts as drift
    
    FFTspecROI = np.fft.rfft(partOfMovieForFFT, axis=2)/(safeendframe - safestartframe)
    
    # Find maximum valued frequency
    mfreq = dcbase + np.argmax(abs(FFTspecROI[x-xmin, y-ymin, dcbase:maxFFTc]))
    #shfreq = mfreq + dcbase + np.argmax(abs(fspec[mfreq + dcbase:]))
    #thirdfreq = shfreq + dcbase + np.argmax(abs(fspec[shfreq + dcbase:]))
    print("Fourier components:", mfreq)
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(np.arange(dcbase, dcbase+len(FFTspecROI[x-xmin, y-ymin, dcbase:maxFFTc])), np.log10(abs(FFTspecROI[x-xmin, y-ymin, dcbase:maxFFTc]))) # Give correct "x"axis
    plt.title("Fourier components of interest")
    plt.xlabel("Fourier component")
    plt.ylabel("Amplitude? (log arb.u.)")
    
    if save_automatically:
        figurename = "region_high/FFTspectrum"
        plotted = FFTspecROI
        customSave(figurename, plotted)
    plt.show()
    
    
    ffpwidth = 3 #estimated half width of the Fourier peak in the power spectrum
    first_harmonic = np.sum(np.abs(FFTspecROI[:, :, mfreq-ffpwidth:mfreq+ffpwidth+1]), axis=2)
    
    fig = plt.figure(figsize=(18,12))
    
    ax1 = plt.subplot(221)
    im1 = ax1.imshow(np.transpose(first_harmonic), cmap='Greys_r', extent=(xmin,xmax,ymin,ymax))
    plt.title("First harmonic")
    plt.xlabel("x")
    plt.ylabel("y")
    clb = plt.colorbar(im1, ax = ax1)
    clb.set_label('(arb.u.)')
    
    ax2 = plt.subplot(222)
    im2 = ax2.imshow(np.transpose(np.log10(first_harmonic)), cmap='Greys_r', extent=(xmin,xmax,ymin,ymax))
    plt.title("Logarithm of First harmonic")
    plt.xlabel("x")
    plt.ylabel("y")
    clb = plt.colorbar(im2, ax = ax2)
    clb.set_label('log (arb.u.)')
    
    mean_img_SOI = mean_img[xmin:xmax,ymin:ymax]
    
    ax3 = plt.subplot(223)
    im3 = ax3.imshow(np.transpose(np.log10(mean_img_SOI)), extent=(xmin,xmax,ymin,ymax))
    plt.title("Logarithm of Mean Image")
    plt.xlabel("x")
    plt.ylabel("y")
    clb = plt.colorbar(im3, ax = ax3)
    clb.set_label('log (arb.u.)')
    
    ax4 = plt.subplot(224)
    im4 = ax4.imshow(np.transpose(np.log10(first_harmonic)), extent=(xmin,xmax,ymin,ymax))
    plt.title("First harmonic divided by Mean Image")
    plt.xlabel("x")
    plt.ylabel("y")
    clb = plt.colorbar(im4, ax = ax4)
    clb.set_label('(arb.u.)')
    
    if save_automatically:
        figurename = "region_high/First_harmonic"
        plotted = first_harmonic
        customSave(figurename, plotted)
    plt.show()
    
    
    
    #%%
    # Looking at the phase:
    
    phase = np.angle(FFTspecROI[:,:, mfreq])/np.pi
    
    fig = plt.figure(figsize=(12,6))
    gs = plt.GridSpec(2, 2, width_ratios=[3, 1], figure=fig)
    
    ax1 = fig.add_subplot(gs[:,0])
    im1 = ax1.imshow(np.transpose(phase), cmap='hsv', extent=(xmin,xmax,ymin,ymax))
    plt.title("Phase")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Custom colormap: 
    # NOTE: not usefull yet, but when I want to add amplitude to these colours it will be, probably.
    ax2 = fig.add_subplot(gs[:,1], projection='polar')
    r = np.linspace(0.01,1,num=100)
    theta = np.linspace(0,2*np.pi,num=200, endpoint=False)
    rs, thetas = np.meshgrid(r, theta)
    colors = thetas
    ax2.scatter(thetas, rs, c=colors, s=0.5+20*r**2, cmap='hsv', alpha=1)
    ax2.set_yticklabels([])
    ax2.grid(False)
    ax2.set_rmax(1)
    plt.xlabel("phase")
    
    if save_automatically:
        figurename = "region_high/phase"
        plotted = phase
        customSave(figurename, plotted)
    plt.show()
    
    
    #%%
    # Oscillations for SOI after summing around spot with highest harmonic component
    
    # Spot halfwidths:
    wx, wy = 10, 5
    
    mx, my = np.unravel_index(np.argmax(first_harmonic), first_harmonic.shape)
    print("Max spot at:", mx + xmin, my + ymin)
    
    avg_spot = np.sum(np.sum(partOfMovieForFFT[mx-wx:mx+wx+1, my-wy:my+wy+1,:]-dark[0,0], axis=0), axis=0)
    
    fig = plt.figure(figsize=(18,6))
    plt.plot(avg_spot,'.')
    plt.title("Bright speckle intensity")
    plt.xlabel("Time (frames)")
    plt.ylabel("Signal (arb.u.)")
    if save_automatically: 
        plt.savefig(dirName+"bright_spot/AveragedPeriod_SOI_"+filename)
    plt.show()
    
    
    #%%
    # Look at self correlation (not that usefull but we can so why not)
    
    FFTspecFiltered = FFTspecROI[x-xmin, y-ymin, :].copy()
    FFTspecFiltered[:dcbase] = 0
    FFTspecFiltered[maxFFTc:] = 0
    specFiltered = np.fft.irfft(FFTspecFiltered)
    
    # First show specle in time after filtering:
    fig = plt.figure(figsize=(8,6))
    plt.plot(specFiltered)
    plt.title("Spot of interest after filtering some Fourier components")
    plt.xlabel("Time (frames)")
    plt.ylabel("Intensity (arb. u.)")
    if save_automatically:
        figurename = "bright_spot/Intensity_SOI-postFFT"
        plotted = specFiltered
        customSave(figurename, plotted)
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
        figurename = "bright_spot/Self-Correlation-SOI-postFFT"
        plotted = correlationAfterFramesFFT
        customSave(figurename, plotted)
    plt.show()
    
    
    
    #%%
    # Another way to summerize the data is Sum[I(t).sin(wt)] where w is the 
    # modulation frequency in 1/frames and the sum is taken over all images.
    
    def getSumIsinW(partOfMovieForFFTVarPlot, mfreq):
        sumitsinwt = np.zeros_like(partOfMovieForFFTVarPlot[:,:,0])
        for i in range(len(partOfMovieForFFTVarPlot[:,0,0])):
            for j in range(len(partOfMovieForFFTVarPlot[0,:,0])):
                sumitsinwt[i,j] = sum([partOfMovieForFFTVarPlot[i,j,t]*np.sin(1/mfreq*t) for t in range(len(partOfMovieForFFTVarPlot[0,0,:]))])
        return sumitsinwt
    
    sumitsinwt = getSumIsinW(partOfMovieForFFT, mfreq)
    
    fig = plt.figure(figsize=(8,6))
    plt.imshow(np.transpose(sumitsinwt), extent=(xmin,xmax,ymin,ymax))
    plt.title("Sum[I(t).sin(wt)] dt")
    plt.xlabel("x")
    plt.ylabel("y")
    clb = plt.colorbar()
    clb.set_label('(arb.u.)')
    if save_automatically:
        figurename = "region_high/Sum-It-sinwt-dt"
        customSave(figurename, sumitsinwt)
    plt.show()
    
    
    
    
    #%%
    # Save all interesting data:
    
    metadataishthing = np.array([("movienumber", movienumber), 
                                 ("camera_fps", Tframe), 
                                 ("frames_per_signal", FPsignal), 
                                 ("signal_amplitude_V", Asignal), 
                                 ("signal_period_seconds", Tsignal), 
                                 ("n_frames", nf), 
                                 ("mfreq", mfreq), 
                                 ("max_var_pos_x", max_var_pos[0]), 
                                 ("max_var_pos_y", max_var_pos[1]), 
                                 ("max_var_val", varPlot[max_var_pos[0], max_var_pos[1]]), 
                                 ("min_var_pos_x", min_var_pos[0]), 
                                 ("min_var_pos_y", min_var_pos[1]), 
                                 ("min_var_val", varPlot[min_var_pos[0], min_var_pos[1]]) ], dtype='object, double')
    
    metadataishthingfile = dirName+"/data_"+filename+'.npy'
    np.save(metadataishthingfile, metadataishthing)

