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

fdir = r'/Users/sanli/Repositories/Data/PDSM_UnderInvestigation/2019-04-30-Pt/'
mfile = 'NaCl_10mM_m2.npy'

trigdata = np.load(fdir+mfile)

R_series = 1.008
ns = 200

real_period = 3 ## actual period in Hz
expo_time = 0.005 ## exposure time in second

ax1 = plt.subplot(111)
color = 'tab:green'

taxis = np.arange(ns) * expo_time
ax1.set_title("Applied potential and measured current")
ax1.set_xlabel('time (Seconds)')
ax1.set_ylabel('applied potential (Volts)', color=color)  # we already handled the x-label with ax1
ax1.plot(taxis, trigdata[:ns,0], '.',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.plot(taxis, trigdata[:ns,1]/R_series, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('current (mA)' , color=color)
plt.show()


fdir = r'/Users/sanli/Repositories/Data/PDSM_UnderInvestigation/2019-04-30-Pt/analyzed/NaCl_10mM/'
mfile = 'NaCl_10mM_seq2_cycavg.npy'

spots_data = np.load(fdir+mfile)
charge = smooth_curve(spots_data[2,:],2)
dt = 0.5
current = (charge - np.roll(charge,1))/dt
taxis = np.arange(np.size(spots_data,1)) * expo_time

ax1 = plt.subplot(111)
color = 'tab:blue'

ax1.set_title("Intensity response and its derivative")
ax1.set_xlabel('time (Seconds)')
ax1.set_ylabel(r'$\Delta I/I$', color=color)  # we already handled the x-label with ax1
ax1.plot(taxis, spots_data[2,:], '.',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.plot(taxis, current, 'o', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel(r'$1/I dI/dt$' , color=color)
plt.show()
