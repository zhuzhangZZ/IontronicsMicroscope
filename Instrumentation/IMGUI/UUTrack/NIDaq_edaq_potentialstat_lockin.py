# -*- coding: utf-8 -*-
"""
Created on Thursday April 8, 2021
@authorr: Zhu Zhang z.zhang@uu.nl
"""

"""
Use the NI DAQ generate triangle + sine wavefoem for AC cyclic voltammatry,
and read data from edaq and photodiode

Annlog Output0(tri+sine) is sent to potentialstat, 
Analog Output1(sine) is sent to lockin-Amp. to server as a reference signal
AI0 is the voltage from EDaq which is the potential of CV
AI1 is the current from EDaq which is the Redox current of CV 
AI3 is the voltage from FEMTO photodiode which is the potential modulated optical signal
"""

import numpy as np
from PyDAQmx import *
from ctypes import byref
from time import sleep
from scipy import signal
import matplotlib.pyplot as plt
import time
import math


#==============================================================================
# Define the function of signal generetion to send signal out by output channel
#==============================================================================
class StepSinGenerator(Task):
    """ define a signal equal to low frq. triangle wave + high freq. sine wave 
    and a signal of sine reference signal
    freq, ampl: frequency and amplitude of sine wave
    off, period: offset and period of triangle wave
    """
    def __init__(self, freq, ampl, off,period):
        Task.__init__(self)
        # Wavegenerator properties:
        self.sampleRate =  200.0*freq # Samples/second
        self.maxNsamples = 200.0*freq
        # Check wave for invalid values:
        if abs(ampl)+abs(off)>10.0:
            print("Tried to supply a potential over (-)10 V")
            return
        # Make array of wave:
        self.Nsamples = int(self.sampleRate/freq)
        self.wavedata = np.abs(np.linspace(2*ampl, -2*ampl, self.Nsamples, endpoint=False)) - ampl + off
        
        self.t = np.linspace(0, 1, int(self.sampleRate*period), endpoint=False)
        self.sin = ampl*np.sin(2*np.pi *freq*period*self.t)
        self.tri = ampl * signal.sawtooth( 2*np.pi * freq*period*self.t, width=0.5)
        # Setup DAQ:
        read = int32()
        self.wavegenOffsetNCycles = 1
        self.wavegenOffsetStart = -0.25
        self.wavegenOffsetEnd = 0.35
        self.wavegenStep = 0.001 # set to small enough then can generate sawtooth wave; if set it to larger ones e.g. 0.1, the wave will be stepwise
        self.steps = math.ceil((self.wavegenOffsetEnd- self.wavegenOffsetStart)/self.wavegenStep)
        self.offsetArray = np.append(np.linspace(self.wavegenOffsetStart, self.wavegenOffsetEnd, num = self.steps)[:-1], 
                                     np.linspace(self.wavegenOffsetEnd, self.wavegenOffsetStart, num = self.steps )[:-1])
        print(self.offsetArray)
        self.wavegenOffsetArray =  np.repeat(self.offsetArray,int(self.sampleRate*period/len(self.offsetArray)))
        if len(self.wavegenOffsetArray) < self.sampleRate*period:
            self.wavegenOffsetArray = np.append(self.wavegenOffsetArray, \
                self.wavegenOffsetStart*np.ones(int(self.sampleRate*period)-len(self.wavegenOffsetArray)) )
#        self.wavegenOffsetArray = np.tile(np.concatenate((\
#                np.arange(0, self.wavegenOffsetEnd, np.sign(self.wavegenOffsetEnd)*self.wavegenStep), 
#                np.arange(self.wavegenOffsetEnd, self.wavegenOffsetStart, -np.sign(self.wavegenOffsetEnd)*self.wavegenStep), 
#                np.arange(self.wavegenOffsetStart, 0, np.sign(self.wavegenOffsetEnd)*self.wavegenStep) )), self.wavegenOffsetNCycles)
        self.ACStep = self.sin + self.wavegenOffsetArray + off
        """the signal which will be sent to AO0 and AO1 are stacked because I choose the the fillmode of 'DAQmx_Val_GroupByChannel'
        in the WriteAnalogF64 function, whcih means the data is grouped by channel(non-interleaved), check the Interleaving on 
        https://zone.ni.com/reference/en-XX/help/370466AH-01/mxcncpts/interleaving/ for more details"""
        self.AC_ref = np.append(self.ACStep,self.sin/max(self.sin)*0.5, axis=0) #self.sin/max(self.sin)*1.0 is the reference signal to lockin
        self.CreateAOVoltageChan("Dev1/ao0", "", -1.0, 1.0, DAQmx_Val_Volts, None)
        self.CreateAOVoltageChan("Dev1/ao1", "", -1.0, 1.0, DAQmx_Val_Volts, None)
        self.CfgSampClkTiming("", int(self.sampleRate), DAQmx_Val_Rising, DAQmx_Val_ContSamps, int(self.sampleRate))
        # self.sampleRate ----The sampling rate in samples per second per channel
        #self.Num ----The number of samples to acquire or generate for each channel in the task( in every period)   self.num*signal_frq=self.N
        self.WriteAnalogF64(int(self.sampleRate*period),bool32(False),-1,DAQmx_Val_GroupByChannel,self.AC_ref,byref(read),None)
        # self.sampleRate ---The number of samples, per channel, to write
        
#==============================================================================
# Define the function of acquirement to get data from analog input  channel
#==============================================================================
class MeasureTask(Task):
    def __init__(self, freq):
        Task.__init__(self)
        # Create data storages:
        self.samplerate = 200.0*freq
        self.updatadataevery = 10
        self.inputchannelsN = 3
        self.data = np.zeros(self.inputchannelsN*self.updatadataevery)
        self.a = []
        self.CreateAIVoltageChan("Dev1/ai0","Voltage",DAQmx_Val_Diff,-10.0,10.0,DAQmx_Val_Volts,None)# from Edaq waveform
        self.CreateAIVoltageChan("Dev1/ai1","Current",DAQmx_Val_Diff,-10.0,10.0,DAQmx_Val_Volts,None)# from eDaq currents
        self.CreateAIVoltageChan("Dev1/ai3","Lockin",DAQmx_Val_Diff,-10.0,10.0,DAQmx_Val_Volts,None) #from lockin output
        # refer to http://zone.ni.com/reference/en-XX/help/370471AA-01/daqmxcfunc/daqmxcreateaivoltagechan/
        self.CfgSampClkTiming("",self.samplerate,DAQmx_Val_Rising,DAQmx_Val_ContSamps,1000) # 1000.0 is the sampling rate
        # refer to http://zone.ni.com/reference/en-XX/help/370471AA-01/daqmxcfunc/daqmxcfgsampclktiming/
                 
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer, self.updatadataevery, 0)
        # refer to http://zone.ni.com/reference/en-XX/help/370471AM-01/daqmxcfunc/daqmxregistereverynsamplesevent/
        self.AutoRegisterDoneEvent(0)
    def EveryNCallback(self):
        read = int32()
        # Handle data, done every "updatadataevery" measurements
        self.ReadAnalogF64(self.updatadataevery, 10.0, DAQmx_Val_GroupByScanNumber, self.data, self.inputchannelsN*self.updatadataevery, byref(read), None)
        self.a.extend(self.data.tolist())
        return 0 # The function should return an integer
    def DoneCallback(self, status):
        print("Status",status.value)
        return 0 # The function should return an integer        
#==============================================================================
# Define the function to zero the optput channels atfer the measurement is done
#==============================================================================
class ZeroOutput(Task):
    def __init__(self):
        Task.__init__(self)
        self.CreateAOVoltageChan("Dev1/ao0","",-10.0,10.0,DAQmx_Val_Volts,None)
        self.CreateAOVoltageChan("Dev1/ao1","",-10.0,10.0,DAQmx_Val_Volts,None)
        self.CfgSampClkTiming("", int(10), DAQmx_Val_Rising, DAQmx_Val_ContSamps, int(10))
        self.WriteAnalogF64(int(10),bool32(False),-1,DAQmx_Val_GroupByChannel,np.zeros((1,20)),byref(int32()),None)
#        self.WriteAnalogScalarF64(1,10.0, np.zeros((1,1))*0.00, None)
#        self.WriteAnalogScalarF64(1,10.0, 0.0,None)
    
#%%
#start generation        
freq = 100
ampl = 10e-3
off = 0
period = 1
taskG = StepSinGenerator(freq=freq,ampl=ampl,off=off,period=period)
taskG.StartTask()
#%%

#start generation and acquisition
taskG.StopTask()
taskG.ClearTask()


taskG = StepSinGenerator(freq=freq,ampl=ampl,off=off,period=period)
taskM = MeasureTask(freq)
taskG.StartTask()

taskM.StartTask()
time_start=time.clock()
PressStop=input('Acquiring samples continuously. Press Enter to interrupt\n')

#time.sleep(1)
time_stop=time.clock()
runtime=time_stop-time_start

taskG.StopTask()
taskM.StopTask()
taskM.ClearTask()
taskG.ClearTask()

taskZ = ZeroOutput()
taskZ.StartTask()
taskZ.StopTask()
taskZ.ClearTask()


data = np.array(taskM.a).reshape(-1,taskM.inputchannelsN)
#    data = np.array(taskM.a)
print(data.shape)
plt.figure(figsize=(12,6))
plt.plot(data[0:50000,0],'r-')
plt.plot(data[0:50000, 1])
plt.plot(data[0:50000, 2])
          
plt.ylim(-0.5, 0.8)
plt.show()
print('Continously record data for %.3f seconds. \nThe shape of the data is %s.' %(runtime, np.shape(data)))
#%%

#%%    
import csv
from datetime import date

today = date.today()
#csv.write('testfile.csv', data, fmt='%.3f')
print(date.today())
with open('CurrentMeasurement_DC+2V_' + str(date.today()) + '.txt','w', newline='') as myfile:
  wr = csv.writer(myfile, delimiter='\t') #, quoting=csv.QUOTE_ALL)
  wr.writerows(data)
