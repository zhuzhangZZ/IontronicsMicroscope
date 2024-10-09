"""
This script imports the averaged intensity vs potential and creates the equivalent of the  I-V curve
assuming that the derivative of intensity is proportional to the current

first draft written by Sanli 6 may 2019

"""
import numpy as np
import matplotlib.pyplot as plt

def smooth_curve(data, boxcar):
    """

    :param data: data to smooth
    :param boxcar: half
    """
    s = np.copy(data)/boxcar
    for  i in np.arange(1, boxcar):
        s = s + np.roll(data, i)/boxcar
    s = np.roll(s, -int(boxcar/2))
    # plt.plot(data)
    # plt.plot(s)
    # plt.show()

    return s


###Loading data from the recorded intensity

fdir = r'/Users/sanli/Repositories/Data/PDSM_UnderInvestigation/2019-04-30-Pt/analyzed/NaCl_10mM/NaCl_10mM_m2_IV.npy/'
mfile = 'NaCl_10mM_m2_IV.npy'

spots_data = np.load(fdir+mfile)
npoints = int(np.size(spots_data,1)/2)
## first column of data is the applied potential

charge = smooth_curve(spots_data[7,:],100)
dv = 3/1000
current = (charge - np.roll(charge,1))/dv


plt.scatter(-spots_data[0,:npoints], charge[:npoints], c = 'k', marker="D", s=4)
plt.scatter(-spots_data[0,npoints:], charge[npoints:], c = 'r', marker="D", s=4)
plt.xlabel('V vs ITO')
plt.ylabel(r'$\Delta I/I$')
plt.show()

plt.scatter(-spots_data[0,:npoints], charge[:npoints], c = 'k', marker="D", s=4)
plt.scatter(-spots_data[0,npoints:], charge[npoints:], c = 'r', marker="D", s=4)
plt.xlabel('V vs ITO')
plt.ylabel(r'$\Delta I/I$')
plt.show()