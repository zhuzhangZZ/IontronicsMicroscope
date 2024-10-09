"""
    For opening all sorts of data files
    ================================
    Import this and use the classes ImportHISdata or ImportHDF5data to handle data nicer.

    Author: Kevin <k.w.namink@uu.nl>
    Using code adapted from Sebastian Haase <haase@msg.ucsf.edu> for the HIS datatype handeling
    
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
              """
    def __init__(self, filename):
        self.f = h5py.File(filename, 'r')
        self.keylist = []
        print("Data in", filename, ": ", self.getkeys())
        
        
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
            
            


if __name__ == '__main__':
    
    print("Tested and it seems to work as intended.")
    
    data = ImportHDF5data("your_movie.hdf5")
    print(f"data.shape")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

