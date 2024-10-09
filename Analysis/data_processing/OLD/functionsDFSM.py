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

def doBin(fig):
    # Return 2x2 binned version of fig 
    dimx, dimy = fig.shape # Ensure even dimensions (lose one line of pixels if it happens)
    if dimx%2 == 1: dimx -= 1
    if dimy%2 == 1: dimy -= 1
    binned = np.zeros([int(dimx/2), int(dimy/2)])
    for k in range(dimx):
        for l in range(dimy):
            binned[int(k/2),int(l/2)] += fig[k,l]/4
    return binned
            

def timeDependentCorrelation(timedependentspot, FPsignal = 1):
    # Find the time dependence of a spot/pixel
    # Making data circular while assuming a full number of frames per signal
    spotintimecircular = timedependentspot[0:int(int(len(timedependentspot)/FPsignal)*FPsignal)]
    spotintimecircular = np.reshape([spotintimecircular,spotintimecircular],-1)
    correlationafterframes = []
    for i in range(int(len(spotintimecircular)/4)):
        correlationafterframes.append(np.correlate(spotintimecircular[0:len(timedependentspot)],spotintimecircular[i:i+len(timedependentspot)]))
    correlationafterframes = np.array(correlationafterframes)
    return correlationafterframes


def pixelFFTFilter(spottime, lowfilter, highfilter):
    # FFT a spot/pixel, then apply filter, then FFT back and return
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

def plotlogMean(meanfigure, colorbar=True):
    # Creates a logplot correctly for a mean of a figure.
    plt.imshow(np.log10(np.transpose(meanfigure)))
    if colorbar:
        clb = plt.colorbar()
        clb.set_label('(arb.u.)')
    plt.title("Logarithm of Averaged Intensity")
    plt.xlabel("x")
    plt.ylabel("y")

def plotlogMeanlogVar(meanfigure, varfigure):
    # Plot log of mean and var correctly
    ax1 = plt.subplot(211)
    im1 = ax1.imshow(np.log10(np.transpose(meanfigure)))
    plt.title("Logarithm of Averaged Intensity")
    plt.ylabel("y")
    clb = plt.colorbar(im1, ax = ax1)
    clb.set_label('(arb.u.)')
    ax2 = plt.subplot(212)
    im2 = ax2.imshow(np.log10(np.transpose(varfigure)))
    plt.title("Logarithm of VarI/I")
    plt.xlabel("x")
    plt.ylabel("y")
    clb = plt.colorbar(im2, ax = ax2)
    clb.set_label('(arb.u.)')

def plotlogMeanlogVarROI(meanfigure, varfigure, roi):
    # Plot log of mean and var correctly
    ax1 = plt.subplot(211)
    im1 = ax1.imshow(np.log10(np.transpose(meanfigure)))
    plt.title("Logarithm of Averaged Intensity")
    plt.ylabel("y")
    clb = plt.colorbar(im1, ax = ax1)
    clb.set_label('(arb.u.)')
    ax2 = plt.subplot(212)
    im2 = ax2.imshow(np.log10(np.transpose(varfigure)))
    plt.title("Logarithm of VarI/I")
    plt.xlabel("x")
    plt.ylabel("y")
    clb = plt.colorbar(im2, ax = ax2)
    clb.set_label('(arb.u.)')
    
def plotlogMeanlogVarwithROI(meanfigure, varfigure, ROI):
    # Plot log of mean and var with interesting spot and ROI
    xmin, xmax, ymin, ymax = getROIaroundXYinFigure(ROI['x'], ROI['y'], varfigure)
    ax1 = plt.subplot(311)
    im1 = ax1.imshow(np.log10(np.transpose(meanfigure)))
    plt.title("Logarithm of Averaged Intensity")
    plt.ylabel("y")
    clb = plt.colorbar(im1, ax = ax1)
    clb.set_label('(arb.u.)')
    plt.scatter(ROI['x'], ROI['y'], s=1, c='red', marker='o')
    ax2 = plt.subplot(312)
    im2 = ax2.imshow(np.log10(np.transpose(varfigure)), extent=(0, varfigure.shape[0], varfigure.shape[1], 0))
    plt.title("Logarithm of VarI/I")
    plt.xlabel("x")
    plt.ylabel("y")
    clb = plt.colorbar(im2, ax = ax2)
    clb.set_label('(arb.u.)')
    plt.vlines([xmin, xmax], ymin, ymax, color='red')
    plt.hlines([ymin, ymax], xmin, xmax, color='red')
    plt.scatter(ROI['x'], ROI['y'], s=5, c='red', marker='o')
    ax3 = plt.subplot(313)
    im3 = ax3.imshow((np.transpose(varfigure[xmin:xmax, ymin:ymax])), extent=(xmin, xmax, ymax, ymin))
    plt.title("VarI/I of ROI")
    plt.xlabel("x")
    plt.ylabel("y")
    clb = plt.colorbar(im3, ax = ax3)
    clb.set_label('(arb.u.)')

def plotFHandmean(FH, mean, mfreq, ext):
    ax1 = plt.subplot(221)
    im1 = ax1.imshow(np.transpose(FH), cmap='Greys_r', extent=ext)
    plt.title("First harmonic (at Fourier component %d)" %mfreq)
    plt.xlabel("x")
    plt.ylabel("y")
    clb = plt.colorbar(im1, ax = ax1)
    clb.set_label('(arb.u.)')
    
    ax2 = plt.subplot(222)
    im2 = ax2.imshow(np.transpose(np.log10(FH)), cmap='Greys_r', extent=ext)
    plt.title("Logarithm of First harmonic (at Fourier component %d)" %mfreq)
    plt.xlabel("x")
    plt.ylabel("y")
    clb = plt.colorbar(im2, ax = ax2)
    clb.set_label('log (arb.u.)')
    
    ax3 = plt.subplot(223)
    im3 = ax3.imshow(np.transpose(np.log10(mean)), extent=ext)
    plt.title("Logarithm of Mean Image")
    plt.xlabel("x")
    plt.ylabel("y")
    clb = plt.colorbar(im3, ax = ax3)
    clb.set_label('log (arb.u.)')
    
    ax4 = plt.subplot(224)
    im4 = ax4.imshow(np.transpose(np.log10(FH/mean)), cmap='magma', extent=ext)
    plt.title("First harmonic divided by Mean Image (at F. com. %d)" %mfreq)
    plt.xlabel("x")
    plt.ylabel("y")
    clb = plt.colorbar(im4, ax = ax4)
    clb.set_label('(arb.u.)')

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

