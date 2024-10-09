"""
This script imports the measured current through the cell and the applied potential to generate the average I-V curve
the procedure is repeated for all different waveforms applied under same conditions

first draft written by Sanli 20 April 2019

"""
import numpy as np
import matplotlib.pyplot as plt

###Loading data from the recorded waveforms

from PDSM_func import AvgWaveForm as awf

fdir = r'/Users/sanli/Repositories/Data/PDSM_UnderInvestigation/2019-04-18-GNP_ITO_flowcell/'
mfile_stem = '01-buffer_m'

vpp = [1, 2, 3]# Note: waveform amplitude goes from -vpp + vbase to vpp + vbase
real_period = 1.0 #The applied waveform period in Hz
vbase = 0.0
wf = "triangle" # or "square" or "sine" ##use this syntax
# wf_duty = 0.50  # duty cycle of the wave
R_series = 9.96 # serial resistor in kOhm, used for measuring current
halfper = 100
taxis = real_period * np.arange(2.0*halfper) / halfper / 2
fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

for i in range(1,4,1):
    mfile = mfile_stem+ str(i) + '.npy'
    print(mfile)
    trigdata = np.load(fdir+mfile)
    #nf = np.size(trigdata, 0)

    scycles = awf.find_potential_cycles(trigdata[:,0], 2*halfper)
    period = awf.find_nframes_cycle(scycles)
    avg_potential = awf.avg_potential_cycle(trigdata[:,0]-trigdata[:,1], scycles)
    avg_current = awf.avg_potential_cycle(trigdata[:,1], scycles) / R_series  # current in mA
    zero_curr = np.argmin(np.abs(avg_current))
    zero_curr = 0
    acc_charge = np.cumsum(np.roll(avg_current,-zero_curr)) * real_period / halfper / 2 * 1000
    taxis = real_period * np.arange(period) / period

    color = 'k'
    ax1.set_title("Cell potential")
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('potential (V)')  # we already handled the x-label with ax1
    ax1.plot(taxis, avg_potential)

    #ax3.set_title("Transferred charge")
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel(r"charge ($\mu$ C)")  # we already handled the x-label with ax1
    ax3.plot(np.roll(taxis,zero_curr), acc_charge)

    ax2.scatter(trigdata[:, 0] - trigdata[:, 1], trigdata[:, 1] / R_series, color ='grey', s=0.2)
    ax2.scatter(avg_potential[:halfper], avg_current[:halfper], c ='k', marker="D", s=8)
    ax2.scatter(avg_potential[halfper:], avg_current[halfper:], c ='r', marker="D", s=8)
    ax2.set_xlabel('cell potential (Volts)')
    ax2.set_ylabel('current (mA)')

    #ax4.set_title("Charge vs Potential")
    ax4.set_xlabel('cell potential (Volts)')
    ax4.set_ylabel(r"charge ($\mu$ C)")  # we already handled the x-label with ax1
    ax4.plot(np.roll(avg_potential,zero_curr), acc_charge)

outf = fdir + 'analyzed/' + mfile_stem + 'IVcurves.png'
fig.savefig(outf)


plt.show()

