"""
    For opening all sorts of data files
    ================================
    Import this and use the classes ImportHISdata or ImportHDF5data to handle data nicer.

    Author: Kevin <k.w.namink@uu.nl>
    Using code adapted from Sebastian Haase <haase@msg.ucsf.edu> for the HIS datatype handeling
    
    Currently most important function: 
    ImportHDF5data() to import hdf5 data files 
    
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

###############################################################################
#
#           Data import functions
#
###############################################################################

# datatypes and reading for HIS data (adapted from Sebastian Haase):
mmap_shape = None # default: map entire file; change this to handle BIG files
# 64 bytes
dtypeHIS = np.dtype([
    ('magic', 'a2'),
    ('ComLen', np.uint16),
    ('iDX', np.uint16),
    ('iDY', np.uint16),
    ('iX', np.uint16),
    ('iY', np.uint16),
    ('pixType', np.uint16),
    ('numImgs', np.uint32),
    ('numChan', np.uint16),
    ('chan', np.uint16),
    ('timeStamp', np.float64),
    ('marker', np.uint32),
    ('miscinfo', '30i1'),
    ])
hisType2numpyDtype = {
    1: np.uint8,
    2: np.uint16,
    3: np.uint32,
    11: ('RGB', (np.uint8, np.uint8, np.uint8)),
    12: ('RGB', (np.uint16, np.uint16, np.uint16)),
}
#http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
# Simple example - adding an extra attribute to ndarray
class ndarray_inHisFile(np.ndarray):
    def __new__(cls, input_array, hisInfo=None):
        obj = np.asarray(input_array).view(cls)
        obj.HIS = hisInfo
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.HIS = getattr(obj, 'HIS', None)

def readSection(m, offsetSect = 0):
    """
    m: numpy memmap of a file, create like m = np.memmap(fn, shape=None, mode = 'r') with fn the file name
    (it saves running time not to create a map each time)
    offsetSect: offset of first byte of section to be read, for next sections use offset = img.HIS.offsetNext
    """
    offsetComment = offsetSect + 64
    hisHdr = m[offsetSect:offsetComment]
    hisHdr.dtype = dtypeHIS
    try:
        hisHdr = hisHdr[0]
    except IndexError:
        print("End of HIS file reached")
        
    commentLength = hisHdr['ComLen']
    offsetImg = offsetComment + commentLength

    if commentLength:
        hisComment = m[offsetComment:offsetImg]
        hisComment.dtype = '|S%d'%(hisHdr['ComLen'],)
    else:
        hisComment = ('',)
    imgPixDType = hisType2numpyDtype[hisHdr['pixType']]
    imgBytes = int(hisHdr['iDX']) * int(hisHdr['iDY']) * imgPixDType().itemsize
    
    sectEnd = offsetImg + imgBytes

    img = m[offsetImg:sectEnd]
    img.dtype = imgPixDType
    img.shape = hisHdr['iDY'], hisHdr['iDX']

    class hisHeaderInfo:
        hdr = hisHdr
        comment = hisComment[0]  # there is "one" comment per sect
        offsetNext = sectEnd

    img = ndarray_inHisFile(img, hisInfo=hisHeaderInfo)

    return hisHdr, img


class ImportHISdata:
    info = """Lists: datanames, -fig and -data are in the same order. 
              Use ImportHISdata.datadata etc. for using this class."""
    def __init__(self, filename):
        self.datanames = []
        self.datafig = []
        self.datadata = []
        m = np.memmap(filename, shape=None, mode = 'r')
        offset = 0
        for i in range(9):
            try:
                inf, img = readSection(m, offset)
            except:
                break
            offset = img.HIS.offsetNext
            self.datanames.append(inf['timeStamp'])
            self.datafig.append(np.copy(img))
            self.datadata.append(inf['marker'] + inf['iDX'] + inf['iDY'] + inf['iX'] + inf['iY'])


# NOTE: the following function is a bit of a mess.
class ImportHDF5data:
    info = """Use ImportHDF5data.f for full raw data, otherwise certain parts 
              of data can be more easily used using functions relating to 'keys' 
              and show() because it already fills in that key stuff once selected.
            
              How to use:
              >>> data = dfsm.ImportHDF5data('/path/to/file.hdf5')  
              <<< Data in /path/to/file.hdf5 :  ['2019-03', '2019-04', '2019-05']
              <<< Wherein there is :  ['metadata', 'timelapse']
              This initiates the class and print the contents as as two lists.
              These lists can then be used to call data using square brackets
              with zero based integers corresponding to the data wanted. For 
              example to get the timelapse of first movie called '2019-03':
              >>> data[0,1]
              <<< <HDF5 dataset "timelapse": shape (18, 11, 20), type "<i2">
              This can then be used to look at the horizontal 1st line of 
              the 10th frame by calling:
              >>> data[0,1][:,1,10]
              <<< array([113, 107, 113,  99, 103, 103, 101, 107, 109, 102, 109,
                         105,  94, 98, 102, 101,  98, 103], dtype=int16)
              
              There are other ways to use this class, but I think this is the
              easiest one. There are functions in this class that you don't
              have to use, but some could be usefull. (Example: '.show()' will
              remember the last 'first' square brackets used. In the example
              above it would mean 'data[0,1]'.)
              """
    def __init__(self, filename):
        self.f = h5py.File(filename, 'r')
        self.keylist = []
        print("Data in", filename, ": ", self.getkeys())
        self.setkey(0)
        print("Wherein there is : ", self.getkeys())
        self.resetkeys()
        
    def __getitem__(self, key):
        self.resetkeys()
        if type(key) == int:
            temp = self.getkeys()[key]
            self.setkey(temp)
            return self.f[temp]
        if len(key) == 2:
            temp = self.getkeys()[key[0]]
            self.setkey(temp)
            temp2 = self.getkeys()[key[1]]
            self.setkey(temp2)
            return self.f[temp][temp2]
            
    def movieN(self, n):
        if type(n) == int:
            temp = self.getkeys()[n]
            self.setkey(temp)
            temp2 = self.getkeys()[1]
            self.setkey(temp2)
            return self.f[temp][temp2]
    
    def givekeys(self):
        print(self.keylist)
        return self.keylist
        
    def getkeys(self):
        if len(self.keylist) == 0:
            return [key for key in self.f.keys()]
        if len(self.keylist) == 1:
            return [key for key in self.f[self.keylist[0]].keys()]
        if len(self.keylist) == 2:
            return [key for key in self.f[self.keylist[0]][self.keylist[1]].keys()]
        
    def resetkeys(self):
        self.keylist = []
        
    def setkey(self, key):
        if type(key) == int:
            keyn = self.getkeys()[key]
        else:
            keyn = key
        self.keylist.append(keyn)
        
    def show(self, optkey = None):
        if optkey == None:
            if len(self.keylist) == 0:
                return self.f
            if len(self.keylist) == 1:
                return self.f[self.keylist[0]]
            if len(self.keylist) == 2:
                return self.f[self.keylist[0]][self.keylist[1]]
        else:
            if len(self.keylist) == 0:
                return self.f[optkey]
            if len(self.keylist) == 1:
                return self.f[self.keylist[0]][optkey]
            if len(self.keylist) == 2:
                return self.f[self.keylist[0]][self.keylist[1]][optkey]
            
            
            
            
###############################################################################
#
#           Data processing Functions
#
###############################################################################

def DAC_signal_properties(DAC_v, varied_offset=False, noise_factor=3):
    """Finds the properties of the DAC signal data
    output: signal_frames, signal_offset, signal_amplitude, signal_shape
    output is always in arrays to be compatible with varied_offset on.
    """
    def single_wave(DAC_v_part):
        minwavs = 5  # Minimal waves expected, if less it might crash
        ff = np.fft.rfft(DAC_v_part)
        fr = len(DAC_v_part)/(np.max([np.argmax(np.abs(ff[minwavs:])),1])+minwavs)
        start = np.argmax(DAC_v_part[:int(fr)])
        argmaxlist = [start]
        while(argmaxlist[-1]+fr*1.25 < len(DAC_v_part)):
            argmaxlist.append(int(argmaxlist[-1]+fr*0.75)+np.argmax(DAC_v_part[int(argmaxlist[-1]+fr*0.75):int(argmaxlist[-1]+fr*1.25)]))
        signal_shape = 'triangle'
        signal_frames = np.mean(np.diff(argmaxlist))
        signal_offset = np.mean(argmaxlist%signal_frames)
        signal_offset_voltage = np.mean(DAC_v_part[argmaxlist[0]:argmaxlist[-1]])
        signal_amplitude = np.mean(DAC_v_part[argmaxlist]) - signal_offset_voltage
        return signal_frames, signal_offset, signal_amplitude, signal_offset_voltage, signal_shape
    
    if varied_offset:
        # Arbitrary constants:
        noiselength_begin = 10  # Frames of "0" potential at the start
        noisefactor = noise_factor  # Factor to overestimate noise with to have some leeway
        # Use derivative of signal to find regions of constant values.
        ddac = np.ediff1d(DAC_v, to_begin=[DAC_v[0]])
        addac = np.roll(np.abs(ddac), 1) + np.abs(ddac)
        noiselim = np.max(addac[:noiselength_begin])*noisefactor
        masker = addac>noiselim
        # Make returnable arrays:
        result = []
        # Find parts:
        x = np.argmax(masker)
        y = 0
        while(x!=y):
            y = np.argmin(masker[x:]) + x
            if y - x > 10:  # If part long enough, add to set of signals:
                a, b, c, d, e = single_wave(DAC_v[x:y])
                result.append({"period": a, "range": [x, y],
                  "phase": b, "amplitude": c,
                  "offset": d, "shape": e})
            x = np.argmax(masker[y:]) + y
        return result
    else:
        a, b, c, d, e = single_wave(DAC_v)
        result =	[{"period": a, "range": [0, len(DAC_v)],
          "phase": b, "amplitude": c,
          "offset": d, "shape": e}]
        return result


def movie_properties(data, movienumber):
    """Finds properties of the movie signal data
    output: xrange, yrange, startframe, endframe, quicklook
    """
    xrange, yrange, Nmax = data[movienumber,1].shape
    quicklook = np.mean(data[movienumber,1][int(xrange/2)-5:int(xrange/2)+5,int(yrange/2),:], axis=0)
    startframe = 0
    endframe = np.argwhere(quicklook==0)[0,0]
    return xrange, yrange, startframe, endframe, quicklook


def get_image_mean_and_var(data, movienumber, startframe, endframe, maximumframes=5000):
    """ <<NO LONGER USED>>
    Creates a mean image of the movie and a variance of the movie (or a part of it to save time)
    prints a few parameters during exectution
    output: mean_img, var_img_MOD
    """
    def maxlengthcenterframes(startframe, endframe, maximumframes):
        # Changes start and end frame number to keep to a certain maximum length
        movieTimeLength = endframe-startframe
        safeendframe, safestartframe = endframe, startframe
        if movieTimeLength > maximumframes:
            safeendframe = endframe - int((movieTimeLength-maximumframes)/2)
            safestartframe = startframe + int((movieTimeLength-maximumframes)/2)
        return safestartframe, safeendframe
    ss, se = maxlengthcenterframes(startframe, endframe, 5000)
    # Calculate mean with substracting dark field:
    mean_img = np.mean(data[movienumber,1][:,:,ss:se], axis=2)
    dark = np.full(mean_img.shape, 96)  # Measured dark field is 96
    dark[dark>=mean_img] = mean_img[dark>=mean_img] - 1 # Set dark field to be lower than usual where the mean image has lower valued pixels, to fix dividing by zero
    mean_img = mean_img - dark
    # Calculate variance:
    var_img = np.var(data[movienumber,1][:,:,ss:se], axis=2)/mean_img
    print("Max and min of Var I/I:", np.max(var_img), np.min(var_img))
    # Modify variance for some calculations:
    var_img_MOD = var_img.copy()
    var_img_MOD[var_img<1] = 1  # To get rid of saturated pixels that will show sub-shot noise variance
    var_img_MOD[var_img>100] = 1  # To get rid of exceptionally irregular points
    print("Filtered when modifying variance: " + str((np.sum(var_img>100)+np.sum(var_img<1))/var_img.size) + " %")
    return mean_img, var_img_MOD


def movie_signal_offset(movie, signal, searchrange=30):
    """Returns the difference in maximum peaks from the movie and signal
    output: moviesignaloffset
    """
    movie_roi = movie[:searchrange]
    movie_treshold = (np.min(movie_roi) + np.max(movie_roi))/2
    moviemasked = movie_roi>movie_treshold
    signal_roi = signal[:searchrange]
    signal_treshold = (np.min(signal_roi) + np.max(signal_roi))/2
    signalmasked = signal_roi>signal_treshold
    
    fromfront = np.argmax(moviemasked) - np.argmax(signalmasked)
    fromback = np.argmax(signalmasked[::-1]) - np.argmax(moviemasked[::-1])
    moviesignaloffset = int((fromfront + fromback)/2)
    
    if np.sum(movie_roi>movie_treshold) > 10:
        print("Peak in intensity not found, possibly happend earlier")
        moviesignaloffset = -13  # Guess -13 because this was correct when this happend before
    return moviesignaloffset


def create_offset_parameters(moviesignaloffset, movie, signal):
    """Returns movie and signal offsets and maximum of the two lengths
    output: mof, sof, msmax
    """
    if moviesignaloffset>0:
        mof = moviesignaloffset
        sof = 0
    else:
        mof = 0
        sof = - moviesignaloffset
    msmax = min(len(movie),len(signal))
    return mof, sof, msmax


def find_max_I_around_guess(mean_img, x_guess, y_guess, areasize=10):
    """ Finds the maximum intensity pixel in an area around the guessed pixel
    output: x_found, y_found
    Plots the result if asked.
    """
    x1, x2, y1, y2 = getROIaroundXYinFigure(x_guess, y_guess, mean_img, xsize=areasize, ysize=areasize)
    particle_area = mean_img[x1:x2,y1:y2]
    x_found, y_found = np.unravel_index(np.argmax(particle_area), particle_area.shape)
    x_found, y_found = x_found + x1, y_found + y1
    return x_found, y_found


def find_FWHM_for_particle(mean_img, x_found, y_found, x_search=30, y_search=15, plot_it=False):
    """ Finds FWHM from the center of a particle
    Where: xx is an array over the x direction, x_FWHM the full image found values
    for the FWHM locations and x1 the offset where xx starts. (And the same for y.)
    output: xx, x_FWHM, x1, yy, y_FWHM, y1
    """
    x1,x2,y1,y2 = getROIaroundXYinFigure(x_found, y_found, mean_img, xsize=x_search, ysize=y_search)
    xx = mean_img[x1:x2,y_found]
    x_FWHM = x_found-np.argmax(mean_img[x_found::-1,y_found]<np.max(xx)/2), x_found+np.argmax(mean_img[x_found:,y_found]<np.max(xx)/2)
    yy = mean_img[x_found,y1:y2]
    y_FWHM = y_found-np.argmax(mean_img[x_found,y_found::-1]<np.max(yy)/2), y_found+np.argmax(mean_img[x_found,y_found:]<np.max(yy)/2)
    if plot_it:
        plt.figure(figsize=(18,6))
        plt.subplot(121)
        plt.plot(xx, '.', color='xkcd:baby blue', label='x direction')
        plt.scatter(x_FWHM[0]-x1, np.max(xx)/2, marker='>', color='xkcd:blue', label='x FWHM')
        plt.scatter(x_FWHM[1]-x1, np.max(xx)/2, marker='<', color='xkcd:blue')
        plt.plot(yy, '.', color='xkcd:baby pink', label='y direction')
        plt.scatter(y_FWHM[0]-y1, np.max(yy)/2, marker='>', color='xkcd:red', label='y FWHM')
        plt.scatter(y_FWHM[1]-y1, np.max(yy)/2, marker='<', color='xkcd:red')
        plt.title('FWHMaxima') ; plt.xlabel("distance (px)") ; plt.ylabel("intensity")
        plt.legend()
        plt.subplot(122)
        _, x2, _, y2 = getROIaroundXYinFigure(x_found, y_found, mean_img, xsize=30, ysize=16)
        plt.imshow((mean_img[x1:x2,y1:y2]>mean_img[x_found, y_found]/2).T,cmap='Greys')
        plt.scatter(x_found-x1, y_found-y1, color='xkcd:white', label='Below half maximum')
        plt.scatter(x_found-x1, y_found-y1, color='xkcd:black', label='Above half maximum')
        plt.hlines(y_found-y1, x_FWHM[0]-x1,x_FWHM[1]-x1, color='xkcd:red')
        plt.vlines(x_found-x1, y_FWHM[0]-y1,y_FWHM[1]-y1, color='xkcd:red', label='FWHMaxima plotted found values')
        plt.scatter(x_found-x1, y_found-y1, color='xkcd:yellow', label='particle max I found location')
        plt.title('Information on particle size chosen') ; plt.legend()
        plt.xlabel("x distance (px)") ; plt.ylabel("y distance (px)")
        plt.show()
    return xx, x_FWHM, x1, yy, y_FWHM, y1


def correct_particle_drift(drifting, polyorfft='poly', polyorder=4, fftfilter=8):
    """ Correct for drift by applying a polynominal or fourrier filter, but keep the
    average the same.
    output: driftcorrected
    """
    if polyorfft == 'poly':
        n = np.size(drifting, 0)
        t = np.arange(n)
        drift = np.polyfit(t, drifting-np.mean(drifting), polyorder)  # Fit a polynome to the data to use as drift correction
        driftcorrected = (drifting - np.polyval(drift, t))            # The drift corrected particle intensity
    if polyorfft == 'fft':
        fft_for_pm = np.fft.rfft(drifting)          # The polynominal doesn't do a good job here,
        fft_for_pm[1:fftfilter] = 0                 # FFT will do a good job: check "np.plot(pm)"
        driftcorrected = np.fft.irfft(fft_for_pm)   # when comparing, and you'll see.
    return driftcorrected


def cycleaverage_particle_and_DAC(particle, adjusted_signal, adjusted_current, frames_per_signal, signal_offset, accepted_rate_two_sigma_points=0.05, waveform='triangle'):
    """ Average to a single period
    output: i, v, c
    which are the cycle averaged values of resp. particle, adjusted_signal, adjusted_current
    """
    def avg_single_period(signal, period, offset, test_outliers=False, outlier_rate=1):
        """
        :param signal: raw signal
        :param period: real (not rounded) period in frames to average over
        :param offset: offset to start from
        :return: the average cycle
        """
        signal_length = len(signal)
        number_of_cycles = int((signal_length - offset)/period)
        number_of_accepted_cycles = 0
        cycle_length = int(period)
        cycle_sum = np.zeros([cycle_length])
        
        # If the variance of a single cycle is significanly larger than in the full signal something bad happend.
        signal_mean = np.mean(signal)
        signal_variance = np.std(signal) 
        
        for i in range(number_of_cycles):
            cycle_start = int(i*period + offset)
            cycle = signal[cycle_start : cycle_start + cycle_length]
            cycle_outliers = np.sum([cycle > signal_mean + 2*signal_variance]) + np.sum([cycle < signal_mean - 2*signal_variance]) 
            if cycle_outliers/cycle_length < outlier_rate or not test_outliers:  # Test for bad cycles
                cycle_sum = cycle_sum + cycle
                number_of_accepted_cycles = number_of_accepted_cycles + 1
            if test_outliers:
                if cycle_outliers/cycle_length < outlier_rate:
                    plt.plot(np.arange(cycle_start, cycle_start + cycle_length), cycle, color='xkcd:blue')
                else:
                    plt.plot(np.arange(cycle_start, cycle_start + cycle_length), cycle, color='xkcd:red')
        if test_outliers:
            plt.plot([], [], color='xkcd:blue', label="accepted cycle after correction (n=%d)"%(number_of_accepted_cycles))
            plt.plot([], [], color='xkcd:red', label="rejected cycle after correction (n=%d)"%(number_of_cycles-number_of_accepted_cycles))
        return cycle_sum/number_of_accepted_cycles
    
    def avg_single_period_squarewave(signal, square_wave, period, offset, test_outliers=False, outlier_rate=1):
        """ Same as before but correctly handeling square waves.
        :param signal: raw signal
        :param period: real (not rounded) period in frames to average over
        :param offset: offset to start from
        :return: the average cycle
        """
        signal_length = len(signal)
        number_of_cycles = int((signal_length - offset)/period)
        number_of_accepted_cycles = 0
        cycle_length = int(period)
        cycle_sum = np.zeros([cycle_length])
        
        # If the variance of a single cycle is significanly larger than in the full signal something bad happend.
        signal_mean = np.mean(signal)
        signal_variance = np.std(signal) 
        
        for i in range(number_of_cycles):
            findstartfrom = int(i*period + offset - period/8)
            findstartswap = square_wave[findstartfrom:int(findstartfrom+period/4)]>0
            cycle_start = int(findstartfrom + np.argmax(findstartswap[0]!=findstartswap))
            cycle = signal[cycle_start : cycle_start + cycle_length]
            cycle_outliers = np.sum([cycle > signal_mean + 2*signal_variance]) + np.sum([cycle < signal_mean - 2*signal_variance]) 
            if cycle_outliers/cycle_length < outlier_rate or not test_outliers:  # Test for bad cycles
                cycle_sum = cycle_sum + cycle
                number_of_accepted_cycles = number_of_accepted_cycles + 1
            if test_outliers:
                if cycle_outliers/cycle_length < outlier_rate:
                    plt.plot(np.arange(cycle_start, cycle_start + cycle_length), cycle, color='xkcd:blue')
                else:
                    plt.plot(np.arange(cycle_start, cycle_start + cycle_length), cycle, color='xkcd:red')
        if test_outliers:
            plt.plot([], [], color='xkcd:blue', label="accepted cycle after correction (n=%d)"%(number_of_accepted_cycles))
            plt.plot([], [], color='xkcd:red', label="rejected cycle after correction (n=%d)"%(number_of_cycles-number_of_accepted_cycles))
        return cycle_sum/number_of_accepted_cycles
    if waveform == "square":
        i = avg_single_period_squarewave(particle, adjusted_signal, frames_per_signal, signal_offset%frames_per_signal, test_outliers=True, outlier_rate=accepted_rate_two_sigma_points)
        v = avg_single_period_squarewave(adjusted_signal, adjusted_signal, frames_per_signal, signal_offset%frames_per_signal)
        c = avg_single_period_squarewave(adjusted_current, adjusted_signal, frames_per_signal, signal_offset%frames_per_signal)
    else:
        i = avg_single_period(particle, frames_per_signal, signal_offset%frames_per_signal, test_outliers=True, outlier_rate=accepted_rate_two_sigma_points)
        v = avg_single_period(adjusted_signal, frames_per_signal, signal_offset%frames_per_signal)
        c = avg_single_period(adjusted_current, frames_per_signal, signal_offset%frames_per_signal)
    return i, v, c


def doBin(fig):
    """ Return 2x2 binned version of fig """
    dimx, dimy = fig.shape # Ensure even dimensions (lose one line of pixels if it happens)
    if dimx%2 == 1: dimx -= 1
    if dimy%2 == 1: dimy -= 1
    binned = np.zeros([int(dimx/2), int(dimy/2)])
    for k in range(dimx):
        for l in range(dimy):
            binned[int(k/2),int(l/2)] += fig[k,l]/4
    return binned

def pixelFFTFilter(spottime, lowfilter, highfilter):
    """ FFT a spot/pixel, then apply filter, then FFT back and return """
    fspotintimefiltered = np.fft.rfft(spottime)
    fspotintimefiltered[0:lowfilter] = 0
    fspotintimefiltered[highfilter:] = 0
    return np.fft.irfft(fspotintimefiltered)



###############################################################################
#
#           Plotting related functions:
#
###############################################################################
    

def getROIaroundXYinFigure(x, y, figure, xsize = 75, ysize = 75):
    # Find the area around a spot of interest in a picture
    xmin, ymin = 0, 0
    xmax, ymax = figure.shape
    xminf = max([xmin, x-xsize])
    xmaxf = min([xmax, x+xsize])
    yminf = max([ymin, y-ysize])
    ymaxf = min([ymax, y+ysize])
    return xminf, xmaxf, yminf, ymaxf

def getPhaseIntensityColormapped(phase, intensity, ext):
    # Plot the phase (delivered in [-1 to 1]) shown relative to the intensity.
    fig = plt.figure(figsize=(12,6))
    result = np.zeros(phase.shape + (4,))
    cmap = plt.cm.hsv(np.arange(plt.cm.hsv.N))
    icmin = np.min(intensity)
    icscale = 1./(np.max(intensity) - icmin)
    for i in range(len(phase[:,0])):
        for j in range(len(phase[0,:])):
            pc = int((phase[i,j]+1) * 256/2)
            ic = icscale*(intensity[i,j] - icmin)
            result[i,j] = np.sqrt(ic)*cmap[pc]
            result[i,j,3] = cmap[pc,3]
    
    gs = plt.GridSpec(2, 2, width_ratios=[3, 1], figure=fig)
    ax1 = fig.add_subplot(gs[:,0])
    ax1.imshow(result, extent=ext)
    plt.title("Phase")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Custom colormap: 
    ax2 = fig.add_subplot(gs[:,1], projection='polar')
    r = np.linspace(0.01,1,num=100)
    theta = np.linspace(0,2*np.pi,num=200, endpoint=False)
    rs, thetas = np.meshgrid(r, theta)
    colors = np.zeros(thetas.shape + (4,))
    for i in range(len(thetas[:,0])):
        for j in range(len(thetas[0,:])):
            colors[i,j] = (rs[i,j])*cmap[int(thetas[i,j]*256/(2*np.pi))]
    colors = colors.reshape(100*200,-1)
    ax2.scatter(thetas, rs, c=colors, s=0.5+20*r**2, alpha=1)
    ax2.set_yticklabels([])
    ax2.grid(False)
    ax2.set_rmax(1)
    plt.xlabel("phase")
    return result


###############################################################################
#
#           Testing:
#
###############################################################################


if __name__ == '__main__':
    
    print("Tested and it seems to work as intended.")
    
    data = ImportHDF5data("GNP20_insertion_movie.hdf5")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

