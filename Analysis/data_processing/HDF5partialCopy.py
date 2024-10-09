# -*- coding: utf-8 -*-
"""
HDF5partialCopy.py 
Creates a partial copy of a HDF5 file

@author: Kevin Namink <k.w.namink@uu.nl>
"""


import h5py

#%%
# Copy the movie at movienumber to a new file. 
# File appears in working directory

# Configure:
folder = "/media/kevin/My Passport/20190723_Crgrid/"
filename = "Crgrid_cell_NaCl10mM_t4"
movienumber = 3

f = h5py.File(folder+filename+".hdf5", 'r')
m = [key for key in f.keys()][movienumber]


fnew = h5py.File(filename+"_m"+str(movienumber)+".hdf5", 'w')

f.copy(m, fnew)


