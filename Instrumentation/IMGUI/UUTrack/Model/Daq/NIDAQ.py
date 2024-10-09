"""
    UUTrack.View.NIcontrol
    =========================
    Holds all communication necessary with the NI DAQ device. 
    Uses the python package "pydaqmx", limited documentation is available online.
    for the functions details, refer to https://www.ni.com/docs/en-US/bundle/ni-daqmx-c-api-ref/page/daqmxcfunc/daqmxcfgsampclktiming.html
    .. sectionauthor:: Kevin Namink <k.w.namink@uu.nl>
       sectionauthor:: Zhu Zhang <z.zhang@uu.nl>
"""

import numpy as np
from PyDAQmx import *
from ctypes import byref
from time import sleep

class BackgroundVoltageArrayTask(Task):
    def __init__(self, size):
        Task.__init__(self)
        # Create data storages:
        self.updatadataevery = 1
        self.inputchannelsN = 1
        self.backgroundarraysize = size  # Size of "backgroundcorrection" time
        self.phase = 0  # Change to synchronize
        self.clock = 0  # Cycles over the backgroundarraysize
        self.data = np.zeros(self.inputchannelsN*self.updatadataevery)
        self.array = np.zeros(self.inputchannelsN*self.backgroundarraysize)  # Array holding what is needed for the "backgroundcorrection" that actually shows the phase contrast live
        self.CreateAIVoltageChan("Dev1/ai0","Voltage",DAQmx_Val_Diff,-10.0,10.0,DAQmx_Val_Volts,None)
        self.CfgSampClkTiming("PFI1", 200.0, DAQmx_Val_Rising, DAQmx_Val_ContSamps, self.updatadataevery)
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer, self.updatadataevery, 0)
        self.AutoRegisterDoneEvent(0)
    def EveryNCallback(self):
        read = int32()
        # Handle data, done every "updatadataevery" measurements
        self.ReadAnalogF64(self.updatadataevery, 10.0, DAQmx_Val_GroupByScanNumber, self.data, self.inputchannelsN*self.updatadataevery, byref(read), None)
        self.clock += 1
        self.array[(self.phase+self.clock)%self.backgroundarraysize] = self.data
        return 0 # The function should return an integer
    def ChangePhase(self, newphase):
        self.phase = int(newphase)
        return 0 # The function should return an integer
        

class MeasureTask(Task):
    def __init__(self):
        Task.__init__(self)
        # Create data storages:
        self.updatadataevery = 10
        self.inputchannelsN = 3
        self.data = np.zeros(self.inputchannelsN*self.updatadataevery)
        self.a = []
        self.CreateAIVoltageChan("Dev1/ai0","Voltage",DAQmx_Val_Diff,-10.0,10.0,DAQmx_Val_Volts,None)
        self.CreateAIVoltageChan("Dev1/ai1","Current",DAQmx_Val_Diff,-10.0,10.0,DAQmx_Val_Volts,None)
        self.CreateAIVoltageChan("Dev1/ai2","LED",DAQmx_Val_Diff,-10.0,10.0,DAQmx_Val_Volts,None)
        self.CfgSampClkTiming("PFI1", 200.0, DAQmx_Val_Rising, DAQmx_Val_ContSamps, self.updatadataevery)  # use the PFI1 to trigger the DAQ to start the task of AI and AO
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer, self.updatadataevery, 0)
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


class PulseTask(Task):
    def __init__(self):
        Task.__init__(self)
        # Create "on" and "off" arrays:
        self.on = data = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.off = data = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
        # Enable channel for use:
        self.CreateDOChan("Dev1/port0/line0:7", "", DAQmx_Val_ChanForAllLines)
    def pulse(self, fps):
        # Send two short pulses to the LED:
        sleep(5/fps)
        self.WriteDigitalLines(1, 1, 10.0, DAQmx_Val_GroupByChannel, self.on, None, None)
        sleep(4/fps/2)
        self.WriteDigitalLines(1, 1, 10.0, DAQmx_Val_GroupByChannel, self.off, None, None)
        sleep(2/fps)
        self.WriteDigitalLines(1, 1, 10.0, DAQmx_Val_GroupByChannel, self.on, None, None)
        sleep(4/fps/2)
        self.WriteDigitalLines(1, 1, 10.0, DAQmx_Val_GroupByChannel, self.off, None, None)



class TriangleWaveGenerator(Task):
    """ Makes a custom triangle wave task and uses it
    """
    def __init__(self, freq, ampl, off):
        Task.__init__(self)
        # Wavegenerator properties:
        self.sampleRate = 200*freq  # Samples/second
        self.maxNsamples = int(200*freq)
        # Check wave for invalid values:
        if abs(ampl)+abs(off)>10.0:
            print("Tried to supply a potential over (-)10 V")
            return
        # Make array of wave:
        
        self.Nsamples = int(self.sampleRate/freq)
        self.wavedata = np.abs(np.linspace(2*ampl, -2*ampl, self.Nsamples, endpoint=False)) - ampl + off
        
        self.t = np.linspace(0, 1, self.Nsamples, endpoint=False)
        self.sin = ampl*np.sin(2*np.pi *self. t) + off
        self.wavedata = self.sin
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


class NIdaqcontrol():
    def __init__(self, size, parent=None):
        # Reset device
        DAQmxResetDevice("Dev1")
        # Remember parent
        self.parent = parent
        # Create tasks defined above:
        self.measuretask = MeasureTask()
        self.pulsetask = PulseTask()
        self.bgvoltagetask = BackgroundVoltageArrayTask(size)
        # Create variables used for keeping track of saving:
        self.sn = None  # "SaveName"
        self.movieN = 0
        self.bgvoltagetaskON = False
        # Wavegenerator parameters  # Make configable [CHECK], also put analysis file on lab pc
        self.wavegenFreq = 10  # Hz
        self.wavegenAmpl = 2.5  # V        
        self.wavegenLength = 10  # seconds  Time spent on each offset (Not very accurate)
        self.wavegenOffsetNCycles = 1  # 
        self.wavegenOffsetStart = 2.0  # V 
        self.wavegenOffsetEnd = 2.5  # V
        self.wavegenStep = 0.2  # deltaV  Make steps of this size with the offset during the measurement
        # Wavegenerator control parameters 
        self.wavegenOffsetArray = np.arange(self.wavegenOffsetStart, self.wavegenOffsetEnd, self.wavegenStep)  # Make cycles stat at 0 and stop at 0 and cycle (so everything twice
        if len(self.wavegenOffsetArray)==0:
            self.wavegenOffsetArray = np.array([self.wavegenOffsetStart])
        self.wavegenOffsetArrayPos = 0
        self.wavegenTiming = 0  # Time since last update
        self.wavegenRUN = False 
        self.wavegenTask = None
        self.automeasure = False
    
    def start(self, fps):
        # Start tasks:
        self.pulsetask.StartTask()
        self.measuretask.StartTask()
        # Do pulse and stop pulse task:
        self.pulsetask.pulse(fps)
        self.pulsetask.StopTask()
        
    def stop(self, savename = "no_name", saveloc = 'C:\\data\\Kevin\\UUTrack\\Signal'):
        # Stop measure task:
        self.measuretask.StopTask()
        # Correctly make a save file path and name:
        if self.sn != savename:
            self.sn = savename
            self.movieN = 0
        else:
            self.movieN += 1
        savefileandpath = (saveloc+"\\"+savename+"_m"+str(self.movieN))
        # Get and save measured data:
        mdata = np.array(self.measuretask.a).reshape(-1,self.measuretask.inputchannelsN)
        np.save(savefileandpath, mdata)
        # Clear task data
        self.measuretask.data = np.zeros(self.measuretask.inputchannelsN*self.measuretask.updatadataevery)
        self.measuretask.a = []
        
    def togglevoltagetask(self, fps):
        if not self.bgvoltagetaskON:
            self.bgvoltagetask.StartTask()
            self.bgvoltagetaskON = True
        else:
            self.bgvoltagetask.StopTask()
            self.bgvoltagetaskON = False
        
    def togglewavegentask(self, settings):
        if self.wavegenRUN:
            # Stop and clear final task:
            self.wavegenTask.stop()
            self.wavegenTask.clear()
            self.wavegenTiming = 0
            # Set DAQ output to 0:
            task = Task()
            task.CreateAOVoltageChan("Dev1/ao0","",-10.0,10.0,DAQmx_Val_Volts,None)
            task.StartTask()
            task.WriteAnalogScalarF64(1,10.0,0.0,None)
            task.StopTask()
            task.ClearTask()
            # Turn switch and return:
            self.wavegenRUN = False
            # Autostop measurement:
            if self.automeasure != 0:
                self.parent.movieSaveStop()
            return
        # Update settings:
        self.wavegenFreq = settings['frequency']
        self.wavegenAmpl = settings['amplitude']
        self.wavegenLength = settings['time_per_offset']
        self.wavegenOffsetNCycles = int(settings['cycles'])
        self.wavegenOffsetStart = settings['offset_start']
        self.wavegenOffsetEnd = settings['offset_end']
        self.wavegenStep = abs(settings['offset_step'])
        self.automeasure = int(settings['auto_measure'])
        # Autostart measurement:
        if self.automeasure != 0:
            self.parent.movieSave()
        # Create offset array:
        if self.wavegenOffsetStart == self.wavegenOffsetEnd:  # If same valued only do that value once
            self.wavegenOffsetArray = np.array([self.wavegenOffsetStart])
        elif (self.wavegenOffsetStart < 0) == (self.wavegenOffsetEnd > 0):  # If opposite signed start and end at 0:
            self.wavegenOffsetArray = np.tile(np.concatenate((\
                np.arange(0, self.wavegenOffsetEnd, np.sign(self.wavegenOffsetEnd)*self.wavegenStep), 
                np.arange(self.wavegenOffsetEnd, self.wavegenOffsetStart, -np.sign(self.wavegenOffsetEnd)*self.wavegenStep), 
                np.arange(self.wavegenOffsetStart, 0, np.sign(self.wavegenOffsetEnd)*self.wavegenStep) )), self.wavegenOffsetNCycles)
        elif abs(self.wavegenOffsetStart) < abs(self.wavegenOffsetEnd):  # If same signed start and end at lower value:
            self.wavegenOffsetArray = np.tile(np.concatenate((\
                np.arange(self.wavegenOffsetStart, self.wavegenOffsetEnd, np.sign(self.wavegenOffsetEnd)*self.wavegenStep), 
                np.arange(self.wavegenOffsetEnd, self.wavegenOffsetStart, -np.sign(self.wavegenOffsetEnd)*self.wavegenStep) )), self.wavegenOffsetNCycles)
        elif abs(self.wavegenOffsetEnd) < abs(self.wavegenOffsetStart):  # If same signed start and end at lower value:
            self.wavegenOffsetArray = np.tile(np.concatenate((\
                np.arange(self.wavegenOffsetEnd, self.wavegenOffsetStart, np.sign(self.wavegenOffsetStart)*self.wavegenStep), 
                np.arange(self.wavegenOffsetStart, self.wavegenOffsetEnd, -np.sign(self.wavegenOffsetStart)*self.wavegenStep) )), self.wavegenOffsetNCycles)
        
        # Actually start task:
        self.wavegenOffsetArrayPos = len(self.wavegenOffsetArray) - 1
        print("Generating wave: ampl =", self.wavegenAmpl, "freq =", self.wavegenFreq, "offsets =", self.wavegenOffsetArray)
        self.wavegenTask = TriangleWaveGenerator(self.wavegenFreq, self.wavegenAmpl,  self.wavegenOffsetArray[self.wavegenOffsetArrayPos])
        self.wavegenTask.start()
        self.wavegenRUN = True
        
    def wavegeneratorcontrol(self, time):
        if not self.wavegenRUN:
            return
        elif self.wavegenTiming < self.wavegenLength:
            self.wavegenTiming += time
        elif self.wavegenOffsetArrayPos > 0:
            self.wavegenTask.stop()
            self.wavegenTask.clear()
            self.wavegenTiming = 0
            self.wavegenOffsetArrayPos -= 1
            self.wavegenTask = TriangleWaveGenerator(self.wavegenFreq, self.wavegenAmpl, self.wavegenOffsetArray[self.wavegenOffsetArrayPos])
            
            self.wavegenTask.start()
        else:
            self.togglewavegentask(None)

    def resetDAQ(self):
        self.bgvoltagetask.StopTask()
        DAQmxResetDevice("Dev1")

if __name__ == '__main__':
    # For testing
    import matplotlib.pyplot as plt
    import time

    if False:
        task = MeasureTask()
        pulsetask = PulseTask()

        task.StartTask()
        pulsetask.StartTask()
        for i in range(10):
            time.sleep(0.5)
            pulsetask.pulse(200)
        

        pulsetask.StopTask()
        pulsetask.ClearTask()
        task.StopTask()
        task.ClearTask()
        

        data = np.array(task.a).reshape(-1,task.inputchannelsN)
        plt.plot(data[:,0])
        plt.show()
        plt.plot(data[:,1])
        plt.show()
        plt.plot(data[:,2])
        plt.show()

    if True:
        task = MeasureTask()
        potentialtask = WaveGeneratorTask()
        
        task.StartTask()
        
        potentialtask.start()
        sleep(10)
        potentialtask.stop()
        
        task.StopTask()
    
    
        data = np.array(task.a).reshape(-1,task.inputchannelsN)
        plt.plot(data[:,0])
        plt.show()
        
        task.ClearTask()
        potentialtask.ClearTask()









