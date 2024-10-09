# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:02:22 2023

@author: ZhuZhang zhuzhang101@gmail.com
"""

# import numpy as np
# import sys,os
# # photometricsDriverPath = 'C:\PyVCAM-master\photometrics\src'
# photometricsDriverPath = 'C:\ZhuZhang\Iontronics4BSI\IontronicsMicroscope\Instrumentation\IMGUI\\UUTrack\Controller\devices\photometrics\src'

# sys.path.append(photometricsDriverPath)
# from pyvcam import pvc 
# from pyvcam.camera import Camera  


import numpy as np
from pint import UnitRegistry
import logging
import warnings
from typing import Tuple
from log import get_logger, log_to_file, log_to_screen

import time
import sys,os
currentcwd = os.getcwd()
currentcwd = "C:\ZhuZhang\Iontronics4BSI\IontronicsMicroscope\Instrumentation\IMGUI"
photometricsDriverPath = os.path.join(currentcwd, "UUTrack\Controller\devices\photometrics\src")
sys.path.append(photometricsDriverPath)
from pyvcam import pvc
from pyvcam.camera import Camera

from _skeleton import cameraBase
ureg = UnitRegistry()
Q_ = ureg.Quantity
logger = get_logger(__name__)

class camera(cameraBase):
    MODE_CONTINUOUS = 1
    MODE_SINGLE_SHOT = 0
    MODE_EXTERNAL = 2

    def __init__(self,camera):
        super().__init__(camera)
        self.cam_num = camera  # Monitor ID
        # self.camera = BaslerCamera(camera)
        self.running = False
        self.mode = self.MODE_CONTINUOUS

        self.maxWidth = 0
        self.maxHeight = 0
        self.width = None
        self.height = None
        # self.mode = None
        self.X = None
        self.Y = None
        self.fps = 1
        self.friendly_name = None

    def initializeCamera(self):
        """ Initializes the communication with the camera. Get's the maximum and minimum width. It also forces
        the camera to work on Software Trigger.

        .. warning:: It may be useful to integrate other types of triggers in applications that need to
            synchronize with other hardware.

        """
        logger.debug('Initializing Basler Camera')
        try:
            pvc.uninit_pvcam()
            print("Uninitializes the PVCAM library.")
        except:
            pass
        
        pvc.init_pvcam()
        # self.camera = next(Camera.detect_camera())  # Use generator to find first camera.

        devices = Camera.get_available_camera_names()
        if len(devices) == 0:
            raise CameraNotFound('No camera found')

        for device in devices:
            if self.cam_num in device:
                print(device)
                self.camera = Camera.select_camera(device)
                self.camera.open()
                self.friendly_name = device
                self.serialNumber = self.camera.serial_no
                print("camera loaded")

        if not self.camera:
            msg = f'{self.cam_num} not found. Please check your config file and cameras connected'
            logger.error(msg)
            raise CameraNotFound(msg)

        self.modelname = self.camera.chip_name
        logger.info(f'Loaded camera {self.modelname}')
        self.camera.reset_rois()
        self.CCDsize = self.camera.sensor_size
        self.maxWidth = self.CCDsize[0]
        self.maxHeight = self.CCDsize[1]
        # self.maxHeight = self.camera.HeightMax.GetValue()
        # offsetX = self.camera.OffsetX.GetValue()
        offsetX = 0
        offsetY = 0
        width = self.maxWidth
        height = self.maxHeight
        self.X = (offsetX, offsetX + width)
        self.Y = (offsetY, offsetY + height)

        # self.camera.RegisterConfiguration(pylon.SoftwareTriggerConfiguration(), pylon.RegistrationMode_ReplaceAll,
        #                                   pylon.Cleanup_Delete)
        self.setAcquisitionMode(self.MODE_SINGLE_SHOT)

    def getTemp(self): # done
        """"get the sCMOS sensor temperature
        """
        return self.camera.temp
    
    def setTemp(self, temperature):
        """set the sensorr temperature 
        """
        self.camera.temp_setpoint = temperature
        return self.camera.temp_setpoint
    
    def getNframes(self, framesNum, exp_time, time_out): # done
        """"
        get_frame function with cameras current settings in rapid-succession to get a 3D numpy array of pixel data from a single snap image.
        Multiple ROIs are not supported.
        Example:
        Getting a sequence
        # Given that the camera is already opened as openCam
        stack = openCam.get_sequence(8)        # Getting a sequence of 8 frames
        firstFrame = stack[0]       # Accessing 2D frames from 3D stack
        lastFrame = stack[7]
        Parameters:
        num_frames (int): The number of frames to be captured in the sequence.
        Optional: exp_time (int): The exposure time to use.
        Optional: timeout_ms (int): Duration to wait for new frames. Default is WAIT_FOREVER.
        """
        return self.camera.get_sequence(framesNum, exp_time, time_out)

    def triggerCamera(self):
        """Triggers the camera.
        """
        if self.mode == self.MODE_CONTINUOUS:
            self.camera.start_live()
#         self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        elif self.mode == self.MODE_SINGLE_SHOT:
            print("Snap trigger mode")

        # if self.camera.IsGrabbing():
        #     logger.warning('Triggering an already grabbing camera')
        # else:
        #     if self.mode == self.MODE_CONTINUOUS:
        #         self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        #     elif self.mode == self.MODE_SINGLE_SHOT:
        #         self.camera.StartGrabbing(1)
        # self.camera.ExecuteSoftwareTrigger()
    def setAcquisitionMode(self, mode):
        """
        Set the readout mode of the camera: Single or continuous.
        Parameters
        mode : int
        One of self.MODE_CONTINUOUS, self.MODE_SINGLE_SHOT
        """
        logger.info(f'Setting acquisition mode to {mode}')
        if mode == self.MODE_CONTINUOUS:
        #     logger.debug(f'Setting buffer to {self.camera.MaxNumBuffer.Value}')
        #     self.camera.OutputQueueSize = self.camera.MaxNumBuffer.Value
        #     self.camera.AcquisitionMode.SetValue('Continuous')
            
            self.mode = mode
        elif mode == self.MODE_SINGLE_SHOT:
        #     self.camera.AcquisitionMode.SetValue('SingleFrame')
            self.mode = mode

        # self.camera.AcquisitionStart.Execute()

    def getAcquisitionMode(self):
        """Returns the acquisition mode, either continuous or single shot.
        """
        return self.mode


    def acquisitionReady(self):
        """Checks if the acquisition in the camera is over.
        """
        return True

    def checkExposureUnit(self):
        """"check the exposure time unit ms or micro s
        first get the exposure value in unit table
        get the Unit table {'One Millisecond': 0, 'One Microsecond': 1}
        """
        self.exp_res = self.camera.exp_res
        index_table = self.camera.exp_resolutions
        expoUnit = list(index_table.keys())[list(index_table.values()).index(self.exp_res)]
        if expoUnit == "One Millisecond":
            expo_Unit = 'ms'
        elif expoUnit == "One Microsecond":
            expo_Unit = 'us'
        return expo_Unit
    
    def setExposureUnit(self, expoUnit ='ms'):
        """"set the exposure time unit in ms or us, defalt is in ms
        {'One Millisecond': 0, 'One Microsecond': 1}
        """
        if expoUnit == "ms":
            self.camera.exp_res = 0
        elif expoUnit == 'us':
            self.camera.exp_res = 1
        else:
            print(" set exposure failed, please choose the unit from ['ms' and 'us'].The exposure unit remain in %s "%self.checkExposureUnit())
        print("set the exposure unit to %s"%(self. checkExposureUnit()))
        return self.checkExposureUnit()


    def setExposure(self, exposure): # done
        
        """ Set the exposure time of the camera in the current unit
        """
            
        self.camera.exp_time = int(exposure)
        self.exposure = exposure
        print("set the exposure time to %d in unit of %s"%(self.getExposure(),self. checkExposureUnit()))
        return self.getExposure()

    def getExposure(self): # done
        """ Get the exposure time of the camera in the current unit
        """
        self.exposure = float(self.camera.exp_time)
        return self.exposure

    
    def readCamera(self):
        # if not self.camera.IsGrabbing():
        #     raise WrongCameraState('You need to trigger the camera before reading from it')

        if self.mode == self.MODE_SINGLE_SHOT:
            grab = self.camera.get_frame(int(self.exposure),  int(self.exposure + 100))  # (exp_time(int), timeout_ms(int))
            print(grab.shape)
            img = [grab.T]
            # img = grab
            # grab.Release()
        # else:
        #     img = []
        #     num_buffers = 50
        #     logger.debug(f'{50} frames available')
        #     if num_buffers:
        #         img = [None] * num_buffers
        #         for i in range(num_buffers):
        #             self.camera.start_seq(int(self.exposure), num_frames=1)
        #             frame, fps, frame_count = self.camera.poll_frame()
        #             if [frame['pixel_data']]:
        #                 img[i] = frame['pixel_data'].reshape(self.getSize())
                        
        else:
            img = []
            
            num_buffers = 15
            # logger.debug(f'{self.camera.NumQueuedBuffers.Value} frames available')
            if num_buffers:
                img = [None] * num_buffers
                for i in range(num_buffers):
                    try:
                        frame, self.fps, frame_count = self.camera.poll_frame(timeout_ms=int(self.exposure + 100), 
                                                                         oldestFrame=True, copyData = False)
                        # print(self.camera.check_frame_status())
                        frame_array = frame['pixel_data'].T
                        img[i] = frame_array
                        low = np.amin(frame['pixel_data'])
                        high = np.amax(frame['pixel_data'])
                        # average = np.average(frame['pixel_data'])
                        print('Min:{} \tMax:{} \tFrame Rate: {:.1f} \tFrame Count: {:.0f}'
                              .format(low, high, self.fps, frame_count))
                    except:
                        pass
            print(self.camera.check_frame_status())
        
        return img  

    def set_bufferNum(self, bufferNUm):
        # self.camera.MaxNumBuffer = int(bufferNUm)
        self.camera.MaxNumBuffer.SetValue(int(bufferNUm))
        return self.camera.MaxNumBuffer.GetValue()

    def setBinning(self, xbin, ybin): # done
        """
        Sets the binning of the camera if supported. Has to check if binning in X/Y can be different or not, etc.
        :param xbin:
        :param ybin:
        :return:
        """
        if xbin == 1 and ybin == 1:
            self.camera.binning = (1,1)
            # self.maxWidth = self.getSize()[0]
            
        
        elif xbin == 2 and ybin == 2:
            self.camera.binning = (2,2)
            self.maxWidth = int(self.maxWidth/2)
            self.maxHeight = int(self.maxHeight/2)
        else:
            self.camera.binning = (1,1)
            

        return self.camera.binning
    # def set_ROI(self, X: Tuple[int, int], Y: Tuple[int, int]) -> Tuple[int, int]:
    
    def getBinning(self): # done
        """
        get the binning of the camera if supported. Has to check if binning in X/Y can be different or not, etc.
        :return:
        """

        return self.camera.binning    
   
    def setROI(self, X, Y): # done
        """ Set up the region of interest of the camera. Basler calls this the
        Area of Interest (AOI) in their manuals. Beware that not all cameras allow
        to set the ROI (especially if they are not area sensors).
        Both the corner positions and the width/height need to be multiple of 4.
        Compared to Hamamatsu, Baslers provides a very descriptive error warning.

        :param tuple X: Horizontal limits for the pixels, 0-indexed and including the extremes. You can also check
            :mod:`Base Camera <pynta.model.cameras.base_camera>`
            To select, for example, the first 100 horizontal pixels, you would supply the following: (0, 99)
        :param tuple Y: Vertical limits for the pixels.

        """
        width = abs(X[1] - X[0]) + 1
        width = int(width - width % 2)
        x_pos = int(X[0] - X[0] % 2)
        height = int(abs(Y[1] - Y[0]) + 1)
        height = int(height - height % 2)
        y_pos = int(Y[0] - Y[0] % 2)
        logger.info(f'Updating ROI: (x, y, width, height) = ({x_pos}, {y_pos}, {width}, {height})')
        if x_pos + width > self.maxWidth: # self.getSize() can update the ROI size based on the binning setting
            raise CameraException('ROI width bigger than current camera width of %d'%self.maxWidth)
        if y_pos + height > self.maxHeight:
            raise CameraException('ROI height bigger than camera height of %d'%self.maxHeight)

        # First clear the current ROI to avoid many ROIs, since the BSI allows to set 15 ROIs
        self.clear_ROI()
        self.camera.set_roi(x_pos, y_pos, width, height)
        logger.debug(f'Setting width to {width}')
        logger.debug(f'Setting X offset to {x_pos}')
        logger.debug(f'Setting Height to {height}')
        logger.debug(f'Setting Y offset to {y_pos}')
        self.X = (x_pos, x_pos + width)
        self.Y = (y_pos, y_pos + width)
        self.width = self.camera.shape(0)[0]
        self.height = self.camera.shape(0)[1]
        return self.getSize()

    def clear_ROI(self): # done
        """ Resets the ROI to the maximum area of the camera.
        # thiw will return to the RIO to 2048 x 2048, and change the binning size to 1
        if the current binning sie is 2, and then we need to set the binning size back to 2
        """
        self.binning = self.getBinning()
        self.camera.reset_rois() 
        self.setBinning(self.binning[0], self.binning[1])
        
        return self.getSize()
        
    def getSize(self): # done
        """Returns the size in pixels of the image being acquired. This is useful for checking the ROI settings.
        """
        return self.camera.shape(0) # we only set 1 ROI, so ROI_index = 0
    def getSerialNumber(self): # done
        """Returns the serial number of the camera.
        """
        # return self.camera.getModelInfo(self.cam_id)
        return self.camera.serial_no

    def GetCCDWidth(self): # done
        """
        Returns
        The CCD width in pixels
        """
        # return self.camera.max_width
        return self.camera.sensor_size[0]

    def GetCCDHeight(self): # done
        """
        Returns
        The CCD height in pixels
        """
        return self.camera.sensor_size[1]
    def get_internal_fps(self):
        """
        Returns
        The CCD realtime frame rate,
        camera.pix_time  Returns the camera's pixel time, which is the inverse of the speed of the camera.
        """
        return self.fps

    def setMaxFPS(self, FPS):
        """
        Sets the maximum framerate for aquisition.
        :param FPS:
        :return:
        """
        # if FPS <= 1/self.camera.pix_time:
        #     self.camera.AcquisitionFrameRate.SetValue(FPS)
        # else:
        #     self.camera.AcquisitionFrameRate.SetValue(self.camera.ResultingFrameRate.GetValue())
        
        if self.checkExposureUnit() == 'ms':
            fpsMAX = 1000/self.getExposure()
        elif self.checkExposureUnit() == 'us':
            fpsMAX = 1000*1000/self.getExposure()
        
        return fpsMAX
            
    def setFanSpeed(self, value):
        """"set the cooling fan speed
        FAN_SPEED_HIGH = 0
        FAN_SPEED_MEDIUM = 1
        FAN_SPEED_LOW = 2
        FAN_SPEED_OFF = 3
        """
        self.camera.fan_speed = value
        return self.getFanSpeed
    def getFanSpeed(self):
        """"return the fan speed information
        """
        if self.camera.fan_speed ==0:
            fanspeed = "HighSpeed"
        elif self.camera.fan_speed ==1:
            fanspeed = "MiddleSpeed"
        elif self.camera.fan_speed ==2:
            fanspeed = "LowSpeed"
        elif self.camera.fan_speed ==3:
            fanspeed = "WaterCooling"
        return fanspeed
    def get_supported_exp_modes(self):
        """return the suported exposure modes in the camera"""
        return self.camera.exp_modes
    def get_supported_exp_resolution(self):
        return self.camera.exp_resolutions
  
    def set_exp_mode(self, key):
        if key == 'Internal Trigger' or key == "Internal":
            self.camera.exp_mode = 'Internal Trigger'
        elif key == 'Edge Trigger' or key == "Edge":
            self.camera.exp_mode = 'Edge Trigger'
        elif key == 'Trigger first' or key == "first":
            self.camera.exp_mode = 'Trigger first'
            
        else:
            self.camera.exp_mode = 'Internal Trigger'
            
    def get_exp_mode(self):
        """supported exposure modes are:
        {'Internal Trigger': 1792, 'Edge Trigger': 2304, 'Trigger first': 2048}"""
        
        if self.camera.exp_mode == 1792:
            self.expmode = 'Internal Trigger'
        elif self.camera.exp_mode == 2304:
            self.expmode = 'Edge Trigger'
        elif self.camera.exp_mode == 2048:
            self.expmode = 'Trigger first'
        return self.expmode
    
    
    
    def stopAcq(self): # done
        # self.camera.stopAcquisition()
        logger.info('Stopping camera')
        self.camera.finish()
    def stopCamera(self): # done
        """Stops the acquisition and closes the connection with the camera.
        """
        try:
            #Closing the camera
            self.camera.close()
            pvc.uninit_pvcam()
            return True
        except:
            #Monitor failed to close
            return False


if __name__ == "__main__":
    import time

    camera = camera("PM")
    camera.initializeCamera()
    print("Camera is %s" % camera.friendly_name)
    print("binning setting is %d %d"%(camera.getBinning()[0],camera.getBinning()[1]))
    camera.clear_ROI()
    a = camera.getSize()
    camera.setROI([1, 600], [11, 700])
    c = camera.getSize()
    fanspeed = camera.getFanSpeed()
    print("The Fan speed is %s" %fanspeed)
    temperature = camera.getTemp()
    print("Sensor temperature is %s" %temperature)
    new_set = camera.setTemp(-20)
    new_temp = camera.getTemp()
    print("Sensor new set is %s and temperature now is %s" %(new_set,new_temp))
    print('ROI size is %d %d' % (a[0], a[1]))
    print('New ROI size is %d %d' % (c[0], c[1]))
    print("max width is %d " % camera.GetCCDWidth())
    print("max Height is %d " % camera.GetCCDHeight())
    camera.setExposureUnit('ms')
    camera.setExposure(exposure = 4)
    print("exposure time is %.1f in unit of %s" % (camera.getExposure(), camera.checkExposureUnit()))
    b = camera.setBinning(2, 2)
    
    camera.setROI([1, 100], [11, 700])
    print("check the binnin value %d"%camera.getBinning()[0])
    print("x binning size is %d" % (b[0]))
    print("x binning size is %d" % (b[1]))
    print("Original CCD size")
    print(camera.CCDsize)
    print("CCD size after binning")
    print(camera.camera.sensor_size)
    print("new ROI after Binning is %d x %d\n "%(camera.getSize()[0], camera.getSize()[1]))
    camera.setROI([1, 100], [11, 200])
    d = camera.getSize()
    print('New ROI size is %d %d' % (d[0], d[1]))
    print(camera.camera.shape(0))
    print('set binning to %d' %(camera.setBinning(1, 1)[0]))
    print("new roi is ")
    print(camera.camera.shape(0))
    
    print('serialnumber is %s' % (camera.getSerialNumber()))
    print('internal fps is %.1f' % camera.get_internal_fps())
    print("set max fps is %.1f" % camera.setMaxFPS(5))
    print('internal fps is %.1f' % camera.get_internal_fps())
    camera.camera.speed_table_index =1
    camera.camera.gain = 1 
    
    cmaeraspeedTable = camera.camera.port_speed_gain_table
    speed_table =  camera.camera.speed_table_index
    print(cmaeraspeedTable)
    print(speed_table)
    print("exposure modes")
    print(camera.camera.exp_modes)
    print(camera.camera.exp_out_modes)
    
    
    print("bit depth is ---> %d" %camera.camera.bit_depth)
    print("Gain is ---> %d" %camera.camera.gain)
    print("Trigger table is --->%s"%camera.camera.trigger_table)
    print("ADC offset is --->%d" % camera.camera.adc_offset)
    print("exposure resolution index is  --->%s" % camera.camera.exp_res_index)
    
    print("exposure resolution value is  --->%s" % camera.camera.exp_res)
    print("exposure resolution unit is  --->%s" % camera.checkExposureUnit())
    exp_resolution_table = camera.camera.exp_resolutions
    print("exposure resolution table is  --->%s" % camera.camera.exp_resolutions)
    
    camera.setAcquisitionMode(camera.MODE_SINGLE_SHOT)
    d = camera.getAcquisitionMode()
    print('acquisition mode is %s' % d)
    e = camera.get_supported_exp_modes()
    print('supported exposure modes are %s' % e)
    f = camera.get_supported_exp_resolution()
    print(' exposure resloution is %s' % f)
    
    camera.set_exp_mode("Edge Trigger")
    g = camera.get_exp_mode()
    print(' exposure mode is %s' % g)
    
    camera.set_exp_mode("Internal Trigger")
    h = camera.get_exp_mode()
    print(' Now the exposure mode is %s' % h)
    
    camera.triggerCamera()
    shot = camera.readCamera()
    print(camera.getSize())
    print("shot image sum is %s" % np.sum(shot[0]))
    shot2 = camera.readCamera()
    print(camera.getSize())
    print("shot image sum is %s" % np.sum(shot2[0]))
    
    
    camera.setAcquisitionMode(camera.MODE_CONTINUOUS)
    e = camera.getAcquisitionMode()
    print('acquisition mode is %s' % e)
    camera.triggerCamera()
    frames = camera.readCamera()
    # time.sleep(2)
    # print(len(frames))
    print(camera.camera.check_frame_status())
    camera.stopAcq()
    print(camera.camera.check_frame_status())
    camera.stopCamera()
    try:
        pvc.uninit_pvcam()
        print("Uninitializes the PVCAM library.")
    except:
        pass
        
    
    

