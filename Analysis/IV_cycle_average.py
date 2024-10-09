"""
This script imports the measured current through the cell and the applied potential to generate the average I-V curve

first draft written by Sanli 20 April 2019

"""
import numpy as np
import matplotlib.pyplot as plt

###Loading data from the recorded waveforms

from PDSM_func import AvgWaveForm as awf

fdir = r'/Users/sanli/Repositories/Data/PDSM_UnderInvestigation/2019-04-18-GNP_ITO_flowcell/'
mfile = '01-buffer_m3.npy'

trigdata = np.load(fdir+mfile)

# for scope files only the current is measured
# the generate the I-V curve the applied waveform parameter should be filled from the lab-journal
vpp = [1, 2, 3]# Note: waveform amplitude goes from -vpp + vbase to vpp + vbase
vbase = 0.0
wf = "triangle" # or "square" or "sine" ##use this syntax
# wf_duty = 0.50  # duty cycle of the wave
real_period = 1 #The applied waveform period in Hz
R_series = 9.96 # serial resistor in kOhm, used for measuring current

halfper = 100
scycles = awf.find_potential_cycles(trigdata[:,0], 2*halfper)
period = awf.find_nframes_cycle(scycles)
avg_potential = awf.avg_potential_cycle(trigdata[:,0]-trigdata[:,1], scycles)
avg_current = awf.avg_potential_cycle(trigdata[:,1], scycles) / R_series  # current in mA

###Plotting the change of intensity over the sequence
fig = plt.figure(figsize=(12,6))
ax1 = plt.subplot(121)
color = 'tab:green'
#ax1.plot(ref_signal_nomean[:int(nper_plot*wf_period)])

taxis = np.arange(3*period)/period * real_period
plt.title("Applied potential and Measured current")
ax1.set_xlabel('time (Seconds)')
ax1.set_ylabel('applied waveform (Volts)', color=color)  # we already handled the x-label with ax1
ax1.plot(taxis, trigdata[:3*period,0], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.plot(taxis, trigdata[:3*period,1]/R_series, color=color)
#ax2.set_ylabel('current (mA)')

#fig.tight_layout()

ax3 = plt.subplot(122)
ax3.scatter(trigdata[:,0]-trigdata[:,1], trigdata[:,1] / R_series, color = 'grey', s=0.2)
ax3.scatter(avg_potential[:halfper], avg_current[:halfper], c = 'k', marker="D", s=8)
ax3.scatter(avg_potential[halfper:], avg_current[halfper:], c = 'r', marker="D", s=8)
ax3.set_xlabel('applied potential (Volts)')
ax3.set_ylabel('current (mA)')
ax3.yaxis.tick_right()

plt.show()


