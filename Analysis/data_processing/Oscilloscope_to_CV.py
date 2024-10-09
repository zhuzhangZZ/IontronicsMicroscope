"""
Oscilloscope_to_CV.py

Imports Oscilloscope data and creates a Cyclo Voltametogram 

This file only works if the different channels are correctly named and assigned.
This requires rewriting parts of code that look intimidating, but there are some comments.

Author: Kevin Namink

"""
import numpy as np
import matplotlib.pyplot as plt
#import AvgWaveForm as awf

fdir = '/home/kevin/Documents/PDSM_data/20190522_Ferrocene'
# "ALL#####" numbers:
ALL_begin = 0
ALL_end = 7

# File structure inside of ALL#### folders:
# Possibly need changing, currently '..ch1.csv', '..ch2.csv', '..ch4.csv' and '..tek.set' are looked at.
filestructure = [['/ALL%04d/F%04dCH1.CSV'%(i, i), '/ALL%04d/F%04dCH2.CSV'%(i, i), 
                  '/ALL%04d/F%04dCH4.CSV'%(i, i), '/ALL%04d/F%04dTEK.SET'%(i, i)] for i in np.arange(ALL_begin, ALL_end+1)]


#%%
# Import data
set_header_length = 18

# Change file[#] to the correct index in filestructure for all tracks:
    
measurements = []
for file in filestructure:
    data = []  # Vout:
    with open(fdir+file[1]) as inputfile: # Change here
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
    with open(fdir+file[0]) as inputfile:
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
    
measurements = np.array(measurements)


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
m = measurements[1]

plt.plot(m[1][m[3]:m[4],1], m[2][m[3]:m[4],1])
plt.show()
# Where: m[1] selects Vout data
# m[1][:,1] selects only the voltage data of Vout (and : takes all points)
# m[3] and m[4] are resp. the start and end of the interesting data


#%%
# Plot:

plt.figure(figsize=(10,8))
for m in measurements[1:]:  # Select the measurements to plot together.
    v = m[0][m[3]:m[4],1]
    a = m[2][m[3]:m[4],1]
    plt.plot(v, a, label=m[5]+"  sr:%.2lf  Vmax:%.2lf"%(m[6], np.max(v)))
plt.xlabel("V")
plt.ylabel("A")
plt.title("All data")
plt.legend()
plt.savefig(fdir+'/All_data.png') # Save
plt.show()



#%%
# Plot some:

plt.figure(figsize=(10,8))
for m in measurements[[2, 3, 4, 7]]:  # Select the measurements to plot together.
    v = m[0][m[3]:m[4],1]
    a = m[2][m[3]:m[4],1]
    i = np.argmax(a)
    print(v[i])
    plt.plot(v, a, label=m[5]+"  sr:%.2lf  Vmax:%.2lf"%(m[6], np.max(v)))
plt.xlabel("V")
plt.ylabel("A")
plt.legend()
plt.savefig(fdir+'/Ferrocene_voltages.png') # Save
plt.show()


#%%
# Plot over time:


for m in measurements[[2, 3, 4, 7]]:  # Select the measurements to plot.
    
    v = m[0][m[3]:m[4],1]
    a = m[2][m[3]:m[4],1]
    t = m[2][m[3]:m[4],0] - np.min(m[2][m[3]:m[4],0])
    
    fig, ax1 = plt.subplots(figsize=(10,5))
    color = 'xkcd:red'
    ax1.set_xlabel('$t$ (s)', size=18)
    ax1.set_ylabel('$potential$ (V)', size=18, color=color) 
    ax1.plot(t, v, color=color)
    ax1.tick_params(axis='y', labelsize=14, labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'xkcd:green'
    ax2.set_ylabel('$current$ (mA)', size=18, color=color)
    ax2.plot(t, a*1000, color=color)
    ax2.tick_params(axis='y', labelsize=14, labelcolor=color)
    
    plt.title(m[5]+"  sr:%.2lf  Vmax:%.2lf"%(m[6], np.max(v)), size=18)
    ax1.tick_params(axis='x', labelsize=14)
    plt.xlim(np.min(t), np.max(t))
    plt.savefig(fdir+'/Ferrocene_plot_over_time_%s.png'%(m[5])) # Save
    plt.show()




