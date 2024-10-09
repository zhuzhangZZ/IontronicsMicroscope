import numpy as np
import h5py

def sampleHDF(file, key=[], nframes=0, corner=[], selsize=[]):
    """
    :param
    file: HDF5 file containing one or multiple sequences of images
    key: key to specific group in the hdf5 file. If empty, method will take the last sequence in the file
    nframes: desired number of frame in the imported data cube. If 0, method will set it to maximum number of non-empty frames
    corner: lower corner coordinates of the selection window
    selsize: selection window, if empty, all data is read
    :returns
    wf: float data cube of intensity (x,y,time)
"""
    if not key:
        ks = file.keys()
        k = next(iter(ks))
    else:
        k = key
    dset = file[k]['timelapse']  # for the moment, either the key input is properly set assumes the hdf5 file starts with an actual group of type 'timelapse' and not a 'snap'. should be taken care of in __future__
    dsize = dset.shape
    print(f"original dataset of size {dsize} found")

    if nframes == 0 or nframes > dsize[2]:
        nf = dsize[2]
    else:
        nf = nframes

    if not corner or not selsize:
        fovx = dsize[0]
        fovy = dsize[1]
    else:
        fovx = selsize[0]
        fovy = selsize[1]

    data = np.zeros((fovx, fovy, nf))

    if not corner or not selsize:
        data = dset[:, :, :nf]
    else:
        for i in range(nf):
            data[:,:,i] = dset[corner[0]:corner[0]+fovx, corner[1]:corner[1]+fovy, i]
        
    return data

def reviewHDFmetadata(file):
    for k in file.keys():
        print(k)
        for item in f[k].items():
            print(item)
        m = f[k]['metadata'].value
        print(m)



#
# fdir = "/Users/sanli/Repositories/Data/PDSM_UnderInvestigation/20181228_ITO_salt/"
# fpath = fdir + "Video.hdf5"
# f = h5py.File(fpath, 'r')
#
# reviewHDFmetadata(f)