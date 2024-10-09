"""
This script imports a sequence of images taken at multiple cycles of the applied potential and averages all cycles for display

first draft written by Sanli 5 Jan. 2019

for __future__:
* [ ] use actual trigger signal to find the voltage during the cycle
* [ ] add waveform averaging for square waves

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spstat


###Loading data from the waveforms extracted from image sequences

from PDSM_func import AvgWaveForm as awf

fdir = r'/Users/sanli/Repositories/Data/PDSM_UnderInvestigation/20190208_ITO_IVcurves/analyzed/TiO2particle_NaI_2mM_vpp1p6_r2/take4/'
mfile = 'TiO2particle_NaI_2mM_vpp1p6_r2corners150x0x300x192_spots.npy'


spots = np.load(fdir+mfile)
nf = np.size(spots,1)
#nf = 2000
nspots = np.size(spots,0)
print(f"{nspots} spots traced in the region" )

# measurement info to be filled from the lab-journal
vpp = 2.8# Note: waveform amplitude goes from -vpp + vbase to vpp + vbase
real_period = 1 #The applied waveform period in Hz
vbase = 0.0
wf = "triangle" # or "square" or "sine" ##use this syntax
nper_plot = 10
# wf_duty = 0.50  # duty cycle of the wave

# use a reference spot to find the frequency and phase of the waveform, good ref spots are from ITO shiny regions
ref_spot = 4
particle_spot = 0
ref_signal = spots[ref_spot,:nf]
ref_avg = np.mean(spots[ref_spot,:nf])
lag_correction = 0 #extra correction to correct for possible mistake in estimation of lag



wf_period, wf_lag = awf.find_period_peakfit(ref_signal, guess=100, show_plot = True)
#wf_period = 109
wf_lag = wf_lag + lag_correction
halfper = int(wf_period/2)
print(f"ref trace with period {wf_period} and lag {wf_lag}")
if wf == "triangle":
    applied_wf = awf.triangle_wf(nf, wf_period, wf_lag, vpp)
    applied_single_period = awf.triangle_wf(int(wf_period), wf_period, - wf_period/4 -1, vpp)
elif wf == "square":
    applied_wf = awf.square_wf(nf, wf_period, wf_lag, vpp)

ref_signal_nomean = (ref_signal - np.mean(ref_signal))/ref_avg
ref_signal_filtered = awf.fft_flat(ref_signal, wf_period)/ref_avg
ref_signal_single_period = awf.avg_single_period(ref_signal_filtered, wf_period, wf_lag)
taxis = real_period * np.arange(int(nper_plot*wf_period))/ wf_period

#plot the Intensity for other spots based on the V from the reference spot
particle_signal = spots[particle_spot,:nf]
particle_avg = np.mean(spots[particle_spot,:nf])
particle_signal_filtered = awf.fft_flat(particle_signal, wf_period)/particle_avg
particle_signal_single_period = awf.avg_single_period(particle_signal_filtered, wf_period, wf_lag)

###Plotting the change of intensity over the sequence
fig = plt.figure(figsize=(12,6))
ax1 = plt.subplot(121)
color = 'tab:green'
#ax1.plot(ref_signal_nomean[:int(nper_plot*wf_period)])

plt.title("Intensity of a single spot")
ax1.set_xlabel('time (Seconds)')
ax1.set_ylabel('applied potential (Volts)', color=color)  # we already handled the x-label with ax1
ax1.plot(taxis, applied_wf[:int(nper_plot*wf_period)], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:gray'
ax2.plot(taxis, ref_signal_filtered[:int(nper_plot*wf_period)], color=color)
ax2.set_ylabel('relative differential intensity, dI/I')

#fig.tight_layout()

ax3 = plt.subplot(122)
ax3.scatter(applied_wf, ref_signal_filtered, color = 'grey', s=0.2)
ax3.scatter(applied_single_period[:halfper], ref_signal_single_period[:halfper], c = 'k', marker="D", s=8)
ax3.scatter(applied_single_period[halfper:], ref_signal_single_period[halfper:], c = 'r', marker="D", s=8)
ax3.set_xlabel('applied potential (Volts)')
ax3.yaxis.tick_right()

plt.show()
#
outputfile = fdir + mfile.strip('spots.npy') + 'spot' + str(ref_spot) + '_reference.png'
fig.savefig(outputfile)

###Plotting the change of intensity over the sequence
fig2 = plt.figure(figsize=(12,6))
ax1 = plt.subplot(121)
color = 'tab:green'
#ax1.plot(ref_signal_nomean[:int(nper_plot*wf_period)])

plt.title("Intensity of a single spot")
ax1.set_xlabel('time (Seconds)')
ax1.set_ylabel('applied potential (Volts)', color=color)  # we already handled the x-label with ax1
ax1.plot(taxis, applied_wf[:int(nper_plot*wf_period)], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:gray'
ax2.plot(taxis, particle_signal_filtered[:int(nper_plot*wf_period)], color=color)
ax2.set_ylabel('relative differential intensity, dI/I')

#fig.tight_layout()

ax3 = plt.subplot(122)
ax3.scatter(applied_wf, particle_signal_filtered, color = 'grey', s=0.2)
ax3.scatter(applied_single_period[:halfper], particle_signal_single_period[:halfper], c = 'k', marker="D", s=8)
ax3.scatter(applied_single_period[halfper:], particle_signal_single_period[halfper:], c = 'r', marker="D", s=8)
ax3.set_xlabel('applied potential (Volts)')
ax3.yaxis.tick_right()

plt.show()
#
outputfile = fdir + mfile.strip('spots.npy') + 'spot' + str(particle_spot) + '_particle.png'
fig2.savefig(outputfile)

slope, intercept, r_value, p_value, std_err = spstat.linregress(applied_single_period, ref_signal_single_period)
print(f"Int-V curve of slope {slope}, intercept {intercept}, error {std_err}")
