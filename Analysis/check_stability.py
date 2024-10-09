"""
This script imports a sequence of images taken at multiple cycles of the applied potential and averages all cycles for display

first draft written by Sanli 5 Jan. 2019

for __future__:
* [ ] use actual trigger signal to find the voltage during the cycle

"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PDSM_func import hdf_import

###Loading data from HDF%
fdir = "/Users/sanli/Repositories/Data/PDSM_UnderInvestigation/20181228_ITO_salt/"
fpath = fdir + "Video.hdf5"
f = h5py.File(fpath, 'r')
# hdf_import.reviewHDFmetadata(f)

frames = hdf_import.sampleHDF(f, nframes=200)
# np.save("tempdata.npy", frames)

###Working on a selection of data for faster loading
# frames = np.load("tempdata.npy")

nf = np.size(frames,2)
print(nf)

fig = plt.figure(figsize=(12,6))
ax1 = plt.subplot(121)
im = ax1.imshow(frames[:,:,0], cmap='Greys_r', origin ='lower', vmax=5000, vmin=100)
plt.colorbar(im, ax = ax1)
plt.title("Single frame")

dark = np.min(frames[:,:,0])-1
ax2 = plt.subplot(122)
drift = np.std(frames, axis=2)/np.sqrt(np.mean(frames, axis=2)-dark)
im = ax2.imshow(drift, cmap='plasma', origin ='lower', vmax = 2.0, vmin=1.5)
plt.colorbar(im, ax = ax2)
plt.title(r'$\sigma(I)/\sqrt{I}$')


plt.show()