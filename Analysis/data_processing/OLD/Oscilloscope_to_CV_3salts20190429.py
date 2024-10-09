"""
Oscilloscope_to_CV.py

Imports Oscilloscope data and creates a Cyclo Voltametogram 


This file only works if the data is the exact shape as the one it was made for at first.

"""
import numpy as np
import matplotlib.pyplot as plt
import AvgWaveForm as awf

fdir = r'/home/kevin/Documents/PDSM_data/20190429_3Salt_Voltametry'
# "ALL#####" numbers:
ALL_begin = 0
ALL_end = 20

# File structure inside of ALL#### folders:
filestructure = [['/ALL%04d/F%04dCH1.CSV'%(i, i), '/ALL%04d/F%04dCH2.CSV'%(i, i), 
                  '/ALL%04d/F%04dCH3.CSV'%(i, i), '/ALL%04d/F%04dTEK.SET'%(i, i)] for i in np.arange(ALL_begin, ALL_end+1)]


#%%
# Import data
set_header_length = 18
    
measurements = []
for file in filestructure:
    data = []  # Vout:
    with open(fdir+file[0]) as inputfile:
        for line in inputfile:
            data.append(line.strip().split(','))
    metadata_Vout = np.array(data[:set_header_length])
    measured_Vout = np.array([[float(j) for j in i if len(j) > 0] for i in data[set_header_length:]])
    
    data = []  # Vin:
    with open(fdir+file[1]) as inputfile:
        for line in inputfile:
            data.append(line.strip().split(','))
    metadata_Vin = np.array(data[:set_header_length])
    measured_Vin = np.array([[float(j) for j in i if len(j) > 0] for i in data[set_header_length:]])
    
    data = []  # Aout:
    with open(fdir+file[2]) as inputfile:
        for line in inputfile:
            data.append(line.strip().split(','))
    metadata_Aout = np.array(data[:set_header_length])
    measured_Aout = np.array([[float(j) for j in i if len(j) > 0] for i in data[set_header_length:]])
    
    data = []  # settings file:
    with open(fdir+file[3]) as inputfile:
        for line in inputfile:
            data.append(line.strip().split(','))
    
    
    # Find interesting range:
    start = np.argmax(measured_Aout[:,1]>0.99*min(measured_Vout[:,1]))
    start = max(start, np.argmax(measured_Aout[:,1]>0.99*min(measured_Aout[:,1])))
    end = np.argmax(measured_Vout[::-1,1]>0.99*min(measured_Vout[:,1]))
    end = -max(end, np.argmax(measured_Aout[::-1,1]>0.99*min(measured_Aout[:,1])))
    if end == 0:
        end = len(measured_Vout[:,1])
    label = file[3][1:8]
    
    # Find scanrate:
    maxi = max(measured_Vin[start:end,1])
    mini = min(measured_Vin[start:end,1])
    fmax = np.argmax(measured_Vin[start:end,1]>maxi*0.9)
    fmin = np.argmax(measured_Vin[start:end,1]<mini*0.9)
    if(fmax>fmin):
        fmin = np.argmax(measured_Vin[start+fmax+1:end,1]<mini*0.9) + fmax+1
        scanrate = 2*maxi/(measured_Vin[start+fmin,0]-measured_Vin[start+fmax,0])
    else:
        fmax = np.argmax(measured_Vin[start+fmin+1:end,1]>maxi*0.9) + fmin+1
        scanrate = 2*maxi/(measured_Vin[start+fmax,0]-measured_Vin[start+fmin,0])
    
    this_set = [measured_Vout, measured_Vin, measured_Aout, start, end, label, scanrate]
    measurements.append(this_set)

#%%




#%%
# Example figures:

# Where 'measurements[0]' is the data set for the folder '/ALL0000/'
# wherein there are 'Vout', 'Vin', 'Aout', 'start', 'end' and 'label'.
    
# Beware! There are a lot of lists together, starting with "measurements":
# first []  -> which dataset/measurement
# second [] -> which part of the dataset Vin, Vout, Aout, start and end resp. 0 based
# third [a,b]  -> (not for start and end) which datapoint as 'a' and as 'b' at 0 the timestamp and at 1 the data

# Plot the first CV as an example: (no averaging used)
m = measurements[15]

plt.plot(m[1][m[3]:m[4],1], m[2][m[3]:m[4],1])
plt.show()
# Where: m[1] selects Vout data
# m[1][:,1] selects only the voltage data of Vout (and : takes all points)
# m[3] and m[4] are resp. the start and end of the interesting data


# Plot all:
plt.figure(figsize=(10,8))
for m in measurements:
    plt.plot(m[0][m[3]:m[4],1], m[2][m[3]:m[4],1], label=m[5])

plt.xlabel("V")
plt.ylabel("A")
plt.legend()
plt.show()


#%%
# Plot each salt at the different frequencies:

# NaCl for different frequencies:
plt.figure(figsize=(10,8))
# Manually add some information to the labels:
frequencies = [" 0.1Hz", " 1Hz", " 3Hz", " 10Hz", " 30Hz", " 100Hz", " 0.1Hz"]
i = 0
for m in measurements[:7]:  # Select the measurements to plot together.
    plt.plot(m[0][m[3]:m[4],1], m[2][m[3]:m[4],1], label=m[5]+frequencies[i]+" scanrate: %.2lf"%(m[6]))
    i += 1
plt.xlabel("V")
plt.ylabel("A")
plt.title("NaCl")
plt.legend()
plt.savefig(fdir+'/NaCl.png') # Save
plt.show()

# NaBr for different frequencies:
plt.figure(figsize=(10,8))
frequencies = [" 0.1Hz", " 1Hz", " 3Hz", " 10Hz", " 30Hz", " 100Hz", " 0.3Hz"]
i = 0
for m in measurements[7:14]:
    plt.plot(m[0][m[3]:m[4],1], m[2][m[3]:m[4],1], label=m[5]+frequencies[i]+" scanrate: %.2lf"%(m[6]))
    i += 1
plt.xlabel("V")
plt.ylabel("A")
plt.title("NaBr")
plt.legend()
plt.savefig(fdir+'/NaBr.png')
plt.show()

# NaI for different frequencies:
plt.figure(figsize=(10,8))
frequencies = [" 0.1Hz", " 1Hz", " 3Hz", " 10Hz", " 30Hz", " 100Hz", " 0.3Hz"]
i = 0
for m in measurements[14:21]:
    plt.plot(m[0][m[3]:m[4],1], m[2][m[3]:m[4],1], label=m[5]+frequencies[i]+" scanrate: %.2lf"%(m[6]))
    i += 1
plt.xlabel("V")
plt.ylabel("A")
plt.title("NaI")
plt.legend()
plt.savefig(fdir+'/NaI.png')
plt.show()

# Plot the different salts:
plt.figure(figsize=(10,8))
salt = [" NaCl", " NaBr", " NaI"]
i = 0
for m in measurements[0:20:7]:
    plt.plot(m[0][m[3]:m[4],1], m[2][m[3]:m[4],1], label=m[5]+salt[i]+" scanrate: %.2lf"%(m[6]))
    i += 1
plt.xlabel("V")
plt.ylabel("A")
plt.title("Different salts at 0.1Hz")
plt.legend()
plt.savefig(fdir+'/Salts_at_0p1Hz.png')
plt.show()




#%%
# Plot each salt at the different frequencies While averaging:

# I made this extra function because I couldn't figure it out without it:
def guessPeriod_from_max(input_array):  # Estimates the period
    array = input_array.copy()
    array[array<0.9*max(array)] = 0  # Set all values not close to the maximum to 0
    start = np.argmax(array>0) 
    startr = np.argmax(array[start:]==0) + start
    end = np.argmax(array[startr:]>0) + startr
    endr = np.argmax(array[end:]==0) + end
    return endr-startr  # Return the difference between the first two times finding 0 after finding nonzero

# NaCl for different frequencies:
plt.figure(figsize=(10,8))
frequencies = [" 0.1Hz", " 1Hz", " 3Hz", " 10Hz", " 30Hz", " 100Hz", " 0.1Hz"]
i = 0
for m in measurements[:7]:
    # Use awf functions to average and plot each measurement:
    p = awf.find_period_minfit(m[0][m[3]:m[4],1], guessPeriod_from_max(m[1][m[3]:m[4],1]))
    v = awf.avg_single_period(m[0][m[3]:m[4],1], p[0], 0)
    v = np.append(v, v[0])
    a = awf.avg_single_period(m[2][m[3]:m[4],1], p[0], 0)
    a = np.append(a, a[0])
    plt.plot(v, a, label=m[5]+frequencies[i]+" scanrate: %.2lf"%(m[6]))
    i += 1
plt.xlabel("V")
plt.ylabel("A")
plt.title("NaCl averaged result")
plt.legend()
plt.savefig(fdir+'/NaCl_averaged.png')
plt.show()

# NaBr for different frequencies:
plt.figure(figsize=(10,8))
frequencies = [" 0.1Hz", " 1Hz", " 3Hz", " 10Hz", " 30Hz", " 100Hz", " 0.3Hz"]
i = 0
for m in measurements[7:13]:
    p = awf.find_period_minfit(m[0][m[3]:m[4],1], guessPeriod_from_max(m[1][m[3]:m[4],1]))
    v = awf.avg_single_period(m[0][m[3]:m[4],1], p[0], 0)
    v = np.append(v, v[0])
    a = awf.avg_single_period(m[2][m[3]:m[4],1], p[0], 0)
    a = np.append(a, a[0])
    plt.plot(v, a, label=m[5]+frequencies[i]+" scanrate: %.2lf"%(m[6]))
    i += 1
plt.xlabel("V")
plt.ylabel("A")
plt.title("NaBr averaged result")
plt.legend()
plt.savefig(fdir+'/NaBr_averaged.png')
plt.show()

# NaI for different frequencies:
plt.figure(figsize=(10,8))
frequencies = [" 0.1Hz", " 1Hz", " 3Hz", " 10Hz", " 30Hz", " 100Hz", " 0.3Hz"]
i = 0
for m in measurements[13:20]:
    p = awf.find_period_minfit(m[0][m[3]:m[4],1], guessPeriod_from_max(m[1][m[3]:m[4],1]))
    v = awf.avg_single_period(m[0][m[3]:m[4],1], p[0], 0)
    v = np.append(v, v[0])
    a = awf.avg_single_period(m[2][m[3]:m[4],1], p[0], 0)
    a = np.append(a, a[0])
    plt.plot(v, a, label=m[5]+frequencies[i]+" scanrate: %.2lf"%(m[6]))
    i += 1
plt.xlabel("V")
plt.ylabel("A")
plt.title("NaI averaged result")
plt.legend()
plt.savefig(fdir+'/NaI_averaged.png')
plt.show()

# Plot the different salts:
plt.figure(figsize=(10,8))
salt = [" NaCl", " NaBr", " NaI"]
i = 0
for m in measurements[0:20:7]:
    p = awf.find_period_minfit(m[0][m[3]:m[4],1], guessPeriod_from_max(m[1][m[3]:m[4],1]))
    v = awf.avg_single_period(m[0][m[3]:m[4],1], p[0], 0)
    v = np.append(v, v[0])
    a = awf.avg_single_period(m[2][m[3]:m[4],1], p[0], 0)
    a = np.append(a, a[0])
    plt.plot(v, a, label=m[5]+salt[i]+" scanrate: %.2lf"%(m[6]))
    i += 1
plt.xlabel("V")
plt.ylabel("A")
plt.title("Different salts at 0.1Hz averaged result")
plt.legend()
plt.savefig(fdir+'/Salts_at_0p1Hz_averaged.png')
plt.show()


#%%
# Current at fixed voltage (1.5V) vs scan rate
volt = 1.5
# NaCl for different frequencies:
plt.figure(figsize=(10,8))
sr = []
low = []
high = []
for m in measurements[:7]:
    # Use awf functions to average each measurement:
    p = awf.find_period_minfit(m[0][m[3]:m[4],1], guessPeriod_from_max(m[1][m[3]:m[4],1]))
    v = awf.avg_single_period(m[0][m[3]:m[4],1], p[0], 0)
    a = awf.avg_single_period(m[2][m[3]:m[4],1], p[0], 0)
    # Find the currents at the fixed voltage:
    if (v>volt)[0] == False:
        low.append(a[v>volt][-1])
        high.append(a[v>volt][0])
    else:
        high.append(a[v>volt][-1])
        low.append(a[v>volt][0])
    sr.append(m[6])
plt.plot(sr, high, '.',  label="High value")
plt.plot(sr, low, '.', label="Low value")
xx = np.linspace(0,max(sr))
plt.plot(xx, max(low)/max(xx**0.5)*xx**0.5, ':r', label="square root")
plt.plot(xx, max(high)/max(xx**0.5)*xx**0.5, ':r', label="square root")
plt.xlabel("scanrate")
plt.ylabel("A")
plt.title("NaCl A at 1.5V vs scanrate")
plt.legend()
plt.savefig(fdir+'/NaCl_Aat1p5VvsScanrate.png')
plt.show()

# NaBr for different frequencies:
plt.figure(figsize=(10,8))
sr = []
low = []
high = []
for m in measurements[7:13]:
    # Use awf functions to average each measurement:
    p = awf.find_period_minfit(m[0][m[3]:m[4],1], guessPeriod_from_max(m[1][m[3]:m[4],1]))
    v = awf.avg_single_period(m[0][m[3]:m[4],1], p[0], 0)
    a = awf.avg_single_period(m[2][m[3]:m[4],1], p[0], 0)
    # Find the currents at the fixed voltage:
    if (v>volt)[0] == False:
        low.append(a[v>volt][-1])
        high.append(a[v>volt][0])
    else:
        high.append(a[v>volt][-1])
        low.append(a[v>volt][0])
    sr.append(m[6])
plt.plot(sr, high, '.',  label="High value")
plt.plot(sr, low, '.', label="Low value")
xx = np.linspace(0,max(sr))
plt.plot(xx, max(low)/max(xx**0.5)*xx**0.5, ':r', label="square root")
plt.plot(xx, max(high)/max(xx**0.5)*xx**0.5, ':r', label="square root")
plt.xlabel("scanrate")
plt.ylabel("A")
plt.title("NaCl A at 1.5V vs scanrate")
plt.legend()
plt.savefig(fdir+'/NaBr_Aat1p5VvsScanrate.png')
plt.show()

# NaI for different frequencies:
plt.figure(figsize=(10,8))
sr = []
low = []
high = []
for m in measurements[13:20]:
    # Use awf functions to average each measurement:
    p = awf.find_period_minfit(m[0][m[3]:m[4],1], guessPeriod_from_max(m[1][m[3]:m[4],1]))
    v = awf.avg_single_period(m[0][m[3]:m[4],1], p[0], 0)
    a = awf.avg_single_period(m[2][m[3]:m[4],1], p[0], 0)
    # Find the currents at the fixed voltage:
    if (v>volt)[0] == False:
        low.append(a[v>volt][-1])
        high.append(a[v>volt][0])
    else:
        high.append(a[v>volt][-1])
        low.append(a[v>volt][0])
    sr.append(m[6])
plt.plot(sr, high, '.',  label="High value")
plt.plot(sr, low, '.', label="Low value")
xx = np.linspace(0,max(sr))
plt.plot(xx, max(low)/max(xx**0.5)*xx**0.5, ':r', label="square root")
plt.plot(xx, max(high)/max(xx**0.5)*xx**0.5, ':r', label="square root")
plt.xlabel("scanrate")
plt.ylabel("A")
plt.title("NaCl A at 1.5V vs scanrate")
plt.legend()
plt.savefig(fdir+'/NaI_Aat1p5VvsScanrate.png')
plt.show()


#%%
# Current at fixed voltage (-1.5V) vs scan rate
volt = -2.5

# NaCl for different frequencies:
plt.figure(figsize=(10,8))
sr = []
low = []
high = []
for m in measurements[:7]:
    # Use awf functions to average each measurement:
    p = awf.find_period_minfit(m[0][m[3]:m[4],1], guessPeriod_from_max(m[1][m[3]:m[4],1]))
    v = awf.avg_single_period(m[0][m[3]:m[4],1], p[0], 0)
    a = awf.avg_single_period(m[2][m[3]:m[4],1], p[0], 0)
    # Find the currents at the fixed voltage:
    if (v<volt)[0] == False:
        low.append(a[v<volt][-1])
        high.append(a[v<volt][0])
    else:
        high.append(a[v<volt][-1])
        low.append(a[v<volt][0])
    sr.append(m[6])
plt.plot(sr, high, '.',  label="High value")
plt.plot(sr, low, '.', label="Low value")
xx = np.linspace(0,max(sr))
plt.plot(xx, min(low)/max(xx**0.5)*xx**0.5, ':r', label="square root")
plt.plot(xx, min(high)/max(xx**0.5)*xx**0.5, ':r', label="square root")
plt.xlabel("scanrate")
plt.ylabel("A")
plt.title("NaCl A at -2.5V vs scanrate")
plt.legend()
plt.savefig(fdir+'/NaCl_Aat-2p5VvsScanrate.png')
plt.show()

# NaBr for different frequencies:
plt.figure(figsize=(10,8))
sr = []
low = []
high = []
for m in measurements[7:13]:
    # Use awf functions to average each measurement:
    p = awf.find_period_minfit(m[0][m[3]:m[4],1], guessPeriod_from_max(m[1][m[3]:m[4],1]))
    v = awf.avg_single_period(m[0][m[3]:m[4],1], p[0], 0)
    a = awf.avg_single_period(m[2][m[3]:m[4],1], p[0], 0)
    # Find the currents at the fixed voltage:
    if (v<volt)[0] == False:
        low.append(a[v<volt][-1])
        high.append(a[v<volt][0])
    else:
        high.append(a[v<volt][-1])
        low.append(a[v<volt][0])
    sr.append(m[6])
plt.plot(sr, high, '.',  label="High value")
plt.plot(sr, low, '.', label="Low value")
xx = np.linspace(0,max(sr))
plt.plot(xx, min(low)/max(xx**0.5)*xx**0.5, ':r', label="square root")
plt.plot(xx, min(high)/max(xx**0.5)*xx**0.5, ':r', label="square root")
plt.xlabel("scanrate")
plt.ylabel("A")
plt.title("NaCl A at -2.5V vs scanrate")
plt.legend()
plt.savefig(fdir+'/NaBr_Aat-2p5VvsScanrate.png')
plt.show()

# NaI for different frequencies:
plt.figure(figsize=(10,8))
sr = []
low = []
high = []
for m in measurements[13:20]:
    # Use awf functions to average each measurement:
    p = awf.find_period_minfit(m[0][m[3]:m[4],1], guessPeriod_from_max(m[1][m[3]:m[4],1]))
    v = awf.avg_single_period(m[0][m[3]:m[4],1], p[0], 0)
    a = awf.avg_single_period(m[2][m[3]:m[4],1], p[0], 0)
    # Find the currents at the fixed voltage:
    if (v<volt)[0] == False:
        low.append(a[v<volt][-1])
        high.append(a[v<volt][0])
    else:
        high.append(a[v<volt][-1])
        low.append(a[v<volt][0])
    sr.append(m[6])
plt.plot(sr, high, '.',  label="High value")
plt.plot(sr, low, '.', label="Low value")
xx = np.linspace(0,max(sr))
plt.plot(xx, min(low)/max(xx**0.5)*xx**0.5, ':r', label="square root")
plt.plot(xx, min(high)/max(xx**0.5)*xx**0.5, ':r', label="square root")
plt.xlabel("scanrate")
plt.ylabel("A")
plt.title("NaCl A at -2.5V vs scanrate")
plt.legend()
plt.savefig(fdir+'/NaI_Aat-2p5VvsScanrate.png')
plt.show()


#%%
# Max current vs scan rate

# NaCl for different frequencies:
plt.figure(figsize=(10,8))
sr = []
low = []
high = []
for m in measurements[:7]:
    # Use awf functions to average each measurement:
    p = awf.find_period_minfit(m[0][m[3]:m[4],1], guessPeriod_from_max(m[1][m[3]:m[4],1]))
    v = awf.avg_single_period(m[0][m[3]:m[4],1], p[0], 0)
    a = awf.avg_single_period(m[2][m[3]:m[4],1], p[0], 0)
    # Find the currents at the fixed voltage:
    low.append(a[a==min(a)][0])
    high.append(a[a==max(a)][0])
    sr.append(m[6])
plt.plot(np.array(sr)**0.5, high, '.',  label="High value")
plt.plot(np.array(sr)**0.5, low, '.', label="Low value")
xx = np.linspace(0,max(sr))
plt.plot(xx**0.5, max(low)+(min(low)-max(low))/max(xx**0.5)*xx**0.5, ':r', label="theory")
plt.plot(xx**0.5, min(high)+(max(high)-min(high))/max(xx**0.5)*xx**0.5, ':r', label="theory")
plt.xlabel("sqrt(scanrate)")
plt.ylabel("A")
plt.title("NaCl A at 1.5V vs scanrate^0.5")
plt.legend()
plt.savefig(fdir+'/NaCl_AmaxvsScanrate.png')
plt.show()

# NaBr for different frequencies:
plt.figure(figsize=(10,8))
sr = []
low = []
high = []
for m in measurements[7:13]:
    # Use awf functions to average each measurement:
    p = awf.find_period_minfit(m[0][m[3]:m[4],1], guessPeriod_from_max(m[1][m[3]:m[4],1]))
    v = awf.avg_single_period(m[0][m[3]:m[4],1], p[0], 0)
    a = awf.avg_single_period(m[2][m[3]:m[4],1], p[0], 0)
    # Find the currents at the fixed voltage:
    low.append(a[a==min(a)][0])
    high.append(a[a==max(a)][0])
    sr.append(m[6])
plt.plot(np.array(sr)**0.5, high, '.',  label="High value")
plt.plot(np.array(sr)**0.5, low, '.', label="Low value")
xx = np.linspace(0,max(sr))
plt.plot(xx**0.5, max(low)+(min(low)-max(low))/max(xx**0.5)*xx**0.5, ':r', label="theory")
plt.plot(xx**0.5, min(high)+(max(high)-min(high))/max(xx**0.5)*xx**0.5, ':r', label="theory")
plt.xlabel("sqrt(scanrate)")
plt.ylabel("A")
plt.title("NaCl A at 1.5V vs scanrate^0.5")
plt.legend()
plt.savefig(fdir+'/NaBr_AmaxvsScanrate.png')
plt.show()

# NaI for different frequencies:
plt.figure(figsize=(10,8))
sr = []
low = []
high = []
for m in measurements[13:20]:
    # Use awf functions to average each measurement:
    p = awf.find_period_minfit(m[0][m[3]:m[4],1], guessPeriod_from_max(m[1][m[3]:m[4],1]))
    v = awf.avg_single_period(m[0][m[3]:m[4],1], p[0], 0)
    a = awf.avg_single_period(m[2][m[3]:m[4],1], p[0], 0)
    # Find the currents at the fixed voltage:
    low.append(a[a==min(a)][0])
    high.append(a[a==max(a)][0])
    sr.append(m[6])
plt.plot(np.array(sr)**0.5, high, '.',  label="High value")
plt.plot(np.array(sr)**0.5, low, '.', label="Low value")
xx = np.linspace(0,max(sr))
plt.plot(xx**0.5, max(low)+(min(low)-max(low))/max(xx**0.5)*xx**0.5, ':r', label="theory")
plt.plot(xx**0.5, min(high)+(max(high)-min(high))/max(xx**0.5)*xx**0.5, ':r', label="theory")
plt.xlabel("sqrt(scanrate)")
plt.ylabel("A")
plt.title("NaCl A at 1.5V vs scanrate^0.5")
plt.legend()
plt.savefig(fdir+'/NaI_AmaxvsScanrate.png')
plt.show()
