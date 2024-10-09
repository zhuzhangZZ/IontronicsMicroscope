"""
    UUTrack.DAQwavegeneration
    =========================
    Side program to use the DAQ as a wavegenerator. 
    Usefull as a reference for writing new code with the DAQ, however not all code here works.

    .. sectionauthor:: Kevin Namink <k.w.namink@uu.nl>
    
    
"""

import numpy as np
from PyDAQmx import *
from ctypes import byref
from time import sleep

class WaveGeneratorTaskOG(Task):
    def __init__(self):
        Task.__init__(self)
        # Wave properties:
        self.varyingAmplitude = 0.2  # Volt
        self.varyingFreqency = 10. 
        self.scanRangeStart = -0.9  # Volt
        self.scanRangeEnd = 0.  # Volt
        self.scanSpeed = 0.01
        # Enable channel for use:
        self.CreateAOVoltageChan("Dev1/ao0","Potential",-10.0,10.0,DAQmx_Val_Volts,None)
        self.CfgSampClkTiming("",10000.0,DAQmx_Val_Rising,DAQmx_Val_FiniteSamps,1000)
        # Create start/stop variable
        self.looping = False
        self.iterator = 0
        self.sleeptime = 0.001  # second (found from oscilloscope how fast it updates the voltage)
    def loop(self):
        slow=self.scanRangeStart
        while self.looping and slow<=self.scanRangeEnd:
            #sleep(self.sleeptime)
            self.iterator += 1
            fast = (np.abs((self.varyingAmplitude*self.iterator*self.sleeptime*self.varyingFreqency)%(4*self.varyingAmplitude)-2*self.varyingAmplitude)-self.varyingAmplitude) # Varying wave
            slow = self.scanRangeStart + self.scanSpeed*self.iterator*self.sleeptime
            self.WriteAnalogScalarF64(1,10.0,fast+slow,None)
            #self.WriteAnalogScalarF64(1,10.0,self.iterator%2,None)
    def startloop(self):
        self.iterator = 0
        if not self.looping:
            self.looping = True
            self.loop()
    def start(self):
        read = int32()
        n = 1000
        data = np.zeros((n,), dtype=np.float64)
        for i in np.arange(n):
            data[i] = (np.abs((self.varyingAmplitude*i*self.sleeptime*self.varyingFreqency)%(4*self.varyingAmplitude)-2*self.varyingAmplitude)-self.varyingAmplitude) # Varying wave
            data[i] += self.scanRangeStart + self.scanSpeed*i*self.sleeptime
        print(data)
        self.WriteAnalogF64(n,0,10.0,DAQmx_Val_GroupByScanNumber,data,byref(read),None)
    def stop(self):
        self.looping = False
        
        
class WaveGeneratorTaskContineous(Task):
    def __init__(self):
        Task.__init__(self)
        # Wave properties:
        self.varyingAmplitude = 0.2  # Volt
        self.varyingFreqency = 10. # Hz
        self.scanRangeStart = -0.9  # Volt
        self.scanRangeEnd = 0.  # Volt
        self.scanSpeed = 0.1  # Volt/second
        # Enable channel for use:
        self.sampleRate = 1000.0  # Samples/second
        self.nsamples = 1000
        self.sampleset = 0
        self.CreateAOVoltageChan("Dev1/ao0","Potential",-10.0,10.0,DAQmx_Val_Volts,None)
        self.CfgSampClkTiming("",self.sampleRate,DAQmx_Val_Rising,DAQmx_Val_ContSamps,self.nsamples)
        
    def sendsample(self):
        read = int32()
        data = np.zeros((self.nsamples,), dtype=np.float64)
        datat = np.arange(self.sampleset*self.nsamples, (self.sampleset+1)*self.nsamples)/self.sampleRate
        for i, t in enumerate(datat):
            data[i] = (np.abs((self.varyingAmplitude*t*4*self.varyingFreqency)%(4*self.varyingAmplitude)-2*self.varyingAmplitude)-self.varyingAmplitude) # Varying wave
            data[i] += self.scanRangeStart + self.scanSpeed*t
        # Send next set of points
        self.WriteAnalogF64(self.nsamples,0,10.0,DAQmx_Val_GroupByScanNumber,np.copy(data),byref(read),None)
        
    def start(self):
        # Send sets of points and start task:
            
        self.sendsample()
        self.StartTask()
        
        while self.sampleset < 10:
            self.sampleset += 1
            self.sendsample()
            print(self.sampleset)
        
        # Send points until done:
        print("done")
        return 0 # The function should return an integer


class WaveGeneratorTaskOld(Task):
    def __init__(self):
        Task.__init__(self)
        # Wave properties:
        self.varyingAmplitude = 0.2  # Volt
        self.varyingFreqency = 10. # Hz
        self.scanRangeStart = -0.9  # Volt
        self.scanRangeEnd = 0.  # Volt
        self.scanSpeed = 0.1  # Volt/step
        # Enable channel for use:
        self.sampleRate = 1000.0  # Samples/second
        self.nsamples = 1000
        self.sampleset = 0
        self.CreateAOVoltageChan("Dev1/ao0","Potential",-10.0,10.0,DAQmx_Val_Volts,None)
        
    def sendsample(self, offset):
        read = int32()
        data = np.zeros((self.nsamples,), dtype=np.float64)
        datat = np.arange(0, self.nsamples)/self.sampleRate
        for i, t in enumerate(datat):
            data[i] = (np.abs((self.varyingAmplitude*t*4*self.varyingFreqency)%(4*self.varyingAmplitude)-2*self.varyingAmplitude)-self.varyingAmplitude) # Varying wave
            data[i] += offset
        # Send next set of points
        self.CfgSampClkTiming("",self.sampleRate,DAQmx_Val_Rising,DAQmx_Val_FiniteSamps,self.nsamples)
        self.WriteAnalogF64(self.nsamples,0,10.0,DAQmx_Val_GroupByScanNumber,np.copy(data),byref(read),None)
        
    def start(self):
        # Send sets of points and start task:
        i=0
        offset = self.scanRangeStart + i*self.scanSpeed
        print("Wavegenerator offset:",offset)
        self.sendsample(offset)
        self.StartTask()
        self.WaitUntilTaskDone(self.nsamples/self.sampleRate+1)
        while offset < self.scanRangeEnd:
            i += 1
            offset = self.scanRangeStart + i*self.scanSpeed
            print("Wavegenerator offset:",offset)
            self.StopTask()
            self.sendsample(offset)
            self.StartTask()
            self.WaitUntilTaskDone((1+i)*self.nsamples/self.sampleRate+1)
        
        # Send points until done:
        self.StopTask()
        print("Wavegenerator done")
        return 0 # The function should return an integer



class TriangleWaveGenerator(Task):
    """
    """
    def __init__(self, ampl, freq, off):
        Task.__init__(self)
        # Wavegenerator properties:
        self.sampleRate = 1000.0  # Samples/second
        self.maxNsamples = 1000
        # Make array of wave:
        self.Nsamples = int(self.sampleRate/freq)
        self.wavedata = np.abs(np.linspace(2*ampl, -2*ampl, self.Nsamples, endpoint=False)) - ampl + off
        # Setup DAQ:
        read = int32()
        self.CreateAOVoltageChan("Dev1/ao0","Potential",-10.0,10.0,DAQmx_Val_Volts,None)
        self.CfgSampClkTiming("",self.sampleRate,DAQmx_Val_Rising,DAQmx_Val_ContSamps,self.maxNsamples)
        self.WriteAnalogF64(self.Nsamples,0,10.0,DAQmx_Val_GroupByScanNumber,self.wavedata,byref(read),None)
        
    def start(self):
        self.StartTask()
        
    def stop(self):
        self.StopTask()
        
    def clear(self):
        self.ClearTask()





if __name__ == '__main__':
    # For testing
    import matplotlib.pyplot as plt
    
    DAQmxResetDevice("Dev1")
        
    if False:
        potentialtask = WaveGeneratorTasktest()
        x,t = potentialtask.start()
        plt.plot(t, x)
        plt.show()
        x2,t2 = potentialtask.EveryNCallback()
        plt.plot(t, x)
        plt.plot(t2, x2)
        plt.show()
        
    if True:
        wavegen=TriangleWaveGenerator(0.2, 10, -0.4)
        wavegen.start()
        sleep(3)
        wavegen.stop()
        wavegen.clear()
        wavegen=TriangleWaveGenerator(0.2, 10, 0.4)
        wavegen.start()
        sleep(3)
        wavegen.stop()
        wavegen.clear()
        wavegen=TriangleWaveGenerator(0, 10, 0)
        wavegen.start()
        sleep(0.1)
        wavegen.stop()
        wavegen.clear()
    









