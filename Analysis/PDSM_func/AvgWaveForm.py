"""
Functions related to averaging an experimentally measure periodic waveform

Last edit by Sanli on 3 march 2019

experimental conditions:
* degree of periodicity should be quantified
* measurement rate is not an integer multiple of the periodicity
* the exact time base can be

"""

import numpy as np
import matplotlib.pyplot as plt

def gen_noisy_waveform(n, period, noise = 0.0):
    x = np.arange(n)
    ns = noise * np.random.uniform(0, 1, n)
    y = np.floor(np.sin(x*2*np.pi/period+1)) + ns
    return y

def square_wf(n, period, lag, vpp, vb = 0.0):
    """
    :param n: total points in the sequence
    :param period: period in points
    :param lag: phase of index zero, triggered to first up-rise from zero
    :param vpp: wf amplitude from - vpp + vb to vpp + vb
    :param duty: duty cycle  --> not implemented yet only works for duty + 0.5
    :param vb: base voltage

    :return: 1d array of amplitudes
    """
    y = 1.0 * vpp  * np.sign(np.sin((np.arange(n) - lag*1.0) * 2 * np.pi / period)) + vb
    return y

def triangle_wf(n, period, lag, vpp, vb = 0.0):
    """
    generates a waveform based on the input parameters

    :param n: total points in the sequence
    :param period: period in points
    :param lag: phase of index zero, triggered to first up-rise from zero
    :param vpp: wf amplitude from - vpp + vb to vpp + vb
    :param duty: not implemented yet, works only for duty = 0.5
    :param vb: base voltage

    :return: 1d array of amplitudes
    """
    x1 = np.mod(np.arange(n) - period/4 - lag, period) / period - 0.5
    x2 = square_wf(n, period, lag+period/4, vpp = 1, vb = 0)
    y = - 4 * vpp * (x1 * x2 + 0.25) + vb
    y [y< - vpp + vb] = vpp + vb  ##correcting the occasional outliers
    return y


def find_period_fft(signal, guess, show_plot = False):

    # dcbase : fourier index below which counts as drift
    dcbase = np.int(np.size(signal,0)/guess/2)
    fspec = np.fft.rfft(signal)
    pspec = np.abs(fspec) ** 2
    guess_freq = np.argmax(abs(pspec[dcbase:]))
    fsegment = pspec[dcbase:dcbase + 2 * guess_freq]
    mfreq = dcbase + np.sum(fsegment * np.arange(2 * guess_freq)) / np.sum(fsegment)
    if show_plot:
        plt.plot(np.log10(pspec[0:np.int(9*mfreq)]))
        plt.axvline(x=mfreq, linewidth=2, color='r')
        plt.show()
    per = np.size(signal,0)/mfreq
    return per

def find_period_minfit(signal, guess, show_plot = False):
    """
    Estimates the period of the measured waveform by fitting a line to all the minimum positions

    For square-like waves, it is better to find the period using the differential of the measurement
    :param signal:
    :param guess:
    :return: fit parameters [lag, period]
    """
    n = np.size(signal, 0)
    per = find_period_fft(signal, guess)
    #print(n, per)
    ncyc = np.int(n/per)
    x = np.arange(ncyc)
    peaks = 0 * x
    peakval = 0 * x
    for i in x:
        if (i+1)*per < n:
            sec = signal[int(i * per):int((i+1)*per)]
        else:
            sec = signal[int(i * per):]
        smin = np.argmin(sec)
        peakval[i] = sec[smin]
        peaks[i] = int(i * per) + smin
    p = np.polyfit(x, peaks, 1)
    amp = np.mean(peakval) - np.mean(signal)
    if show_plot:
        plt.scatter(x, peaks)
        fit = np.polyval(p, x)
        plt.plot(x, fit)
        plt.show()
        print(f"{ncyc} peaks with periodicity {p[0]} and amplitude {amp}")

    period = p[0]
    lag = p[1] + p[0]/4

    return period, lag


def find_potential_cycles(signal, guess):
    """
    Finds the sequence indices corresponding to the (rising) beginning of each cycle
    works best for clean signals from the signal generator.
    Estimates the period of the measured waveform by fitting a line to all the peak positions

    For square-like waves, it is better to find the period using the differential of the measurement
    :param signal: np array of the measured waveform
    :param guess: approximate periodicity

    :return: indices of cycle start point

    """
    n = np.size(signal,0)
    per = np.int(guess)
    ncyc = int(np.fix(n/per))
    if signal[2]>signal[0]:      ## to deal with cases that the first return point is a max
        pnt = int(np.fix(per/2))
        ncyc = ncyc - 1
    else:
        pnt = 0
    #print('pointer at', pnt)
    x = np.arange(ncyc)
    s = 0 * x
    for i in x:
        sec = signal[pnt:pnt+per]
        if np.size(sec)>=per:
            s[i] = np.argmin(sec) + pnt
        pnt = pnt + per
        if i>1:
            per = s[i] - s[i-1]
        #print('last period =', per)

    return s


def find_nframes_cycle(starts):
    """
    :param scycles: indices of the cycle start; the output of find_potential_cycles

    :return: nframes in a cycle
    """
    ncyc = np.size(starts)
    sycs = starts[1:] - starts[:ncyc-1]
    p = int(np.mean(sycs))

    return p


def avg_potential_cycle(signal, starts):
    """
    calculate the average over many cycles given the start indices

        :param signal: the unaveraged waveform
        :param starts:
        :return:
        """
    n = np.size(signal, 0)
    per = find_nframes_cycle(starts)
    ncyc = np.size(starts,0)
    avg = np.zeros(per)
    for i in np.arange(ncyc):
        avg = avg + signal[starts[i]:starts[i]+per]/ncyc

    return avg

def avg_intensity_cycle(signal, starts):
    """
    calculate the average of intensity variation over many cycles given the start indices
    after correcting for the drift

        :param signal: the unaveraged waveform
        :param starts:
        :return: average cycle
        """

    n = np.size(signal, 0)
    t = np.arange(n)
    p = np.polyfit(t, signal, 3)
    sig_min_drift = signal - np.polyval(p, t)
    pp = find_nframes_cycle(starts)
    ncyc = np.size(starts, 0)
    avg = np.zeros(pp)
    for i in np.arange(ncyc):
        if starts[i] + pp < n:
            cyc = sig_min_drift[starts[i]:starts[i] + pp]
            avg = avg + (cyc - np.mean(cyc)) / ncyc

    return avg


def find_period_peakfit(signal, guess, show_plot = False):
    """
    Estimates the period of the measured waveform by fitting a line to all the peak positions

    For square-like waves, it is better to find the period using the differential of the measurement
    :param signal:
    :param guess:
    :return: fit parameters [lag, period]
    """
    n = np.size(signal, 0)
    per = find_period_fft(signal, guess)
    #print(n, per)
    ncyc = int(np.fix(n/per))
    x = np.arange(ncyc)
    peaks = 0 * x
    peakval = 0 * x
    for i in x:
        if (i+1)*per < n:
            sec = signal[int(i * per):int((i+1)*per)]
        else:
            sec = signal[int(i * per):]
        smax = np.argmax(sec)
        peakval[i] = sec[smax]
        peaks[i] = int(i * per) + smax
    p = np.polyfit(x, peaks, 1)
    amp = np.mean(peakval) - np.mean(signal)
    if show_plot:
        plt.scatter(x, peaks)
        fit = np.polyval(p, x)
        plt.plot(x, fit)
        plt.show()
        print(f"{ncyc} peaks with periodicity {p[0]} and amplitude {amp}")

    period = p[0]
    lag = p[1] - p[0]/4

    return period, lag

def fft_flat(signal, period, dcper = 0.5):
    """

    :param signal: raw signal
    :param period: period of lowest harmonic
    :param dcper: fraction of harmonic period that is put at the filter treshold
    :return: filtered signal
    """
    fspec = np.fft.rfft(signal)
    yy = signal
    n = np.size(signal)
    dcbase = int(dcper * n/period)
    fspec [0:dcbase] = 0 * fspec [0:dcbase]
    y = np.fft.irfft(fspec)
    nff = np.size(y)
    if nff<n: #to make sure filtered waveform is the same length as signal
        yy[0:nff] = y
        yy[nff] = np.mean(y)
    else:
        yy = y
    return yy

def avg_single_period(signal, period, lag):
    """

    :param signal: raw signal
    :param period: period calculated by find_period_pealfit
    :param lag: period calculated by find_period_pealfit
    :return:
    """
    n = np.size(signal)
    ncyc = np.int(n/period)
    intper = np.int(period)
    y = np.zeros([intper])

    for i in range(ncyc-1):
        b = np.int(i * period + lag + period/4)
        sec = signal[b: b+intper]
        y = y + sec/(ncyc-1)

    return y


def find_period_corr(signal, guess):
    corr_response = np.correlate(signal, signal, mode="same")
    cen = np.argmax(corr_response)
    min1 = cen + np.argmin(corr_response[cen:cen+guess])
    max1 = min1 + np.argmax(corr_response[min1:min1+guess])
    perq = np.int((max1 - min1)/2)
    segment = corr_response[max1-perq:max1+perq+1]
    fraction = np.sum(segment * np.arange(-perq,perq+1)) / np.sum(segment)
    #print(cen, min1, max1, fraction)
    per = max1 - cen + fraction
    return per

# sig_len = 1550
# guess_period = 53.3
# ph = guess_period * 0.75
# signal1 = square_wf(sig_len, guess_period, ph, 1, vb = 0)
# signal2 = triangle_wf(sig_len, guess_period, ph, 2.5, vb = 0)
#
# print("period from fft", find_period_fft(signal2, guess_period - 6, show_plot= False))
#
# period, lag = find_period_minfit(signal2, guess_period - 6, show_plot = False)
#
# print("period and cycle lag from minfit", period, lag/period)
#
# period, lag = find_period_peakfit(signal2, guess_period - 6, show_plot = False)
#
# print("period and cycle lag from peakfit", period, lag/period)

##print(find_period_corr(signal1, guess=16))
##plt.plot(np.log(abs(fspec[dcbase:])))
#
# fig = plt.figure(1)
# ax1 = plt.subplot(111)
# ax1.plot(signal1, '.-')
# ax1.plot(signal2, 'r+')
#
# plt.show()