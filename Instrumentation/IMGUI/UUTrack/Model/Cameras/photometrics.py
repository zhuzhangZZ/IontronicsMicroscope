"""
    UUTrack.Model.Cameras.photometrics.py
    ==================================

    Model class for controlling Photometrics BSI cameras via de PyVCAM API.
    This is absed on the official python package PyVCAM, https://github.com/Photometrics/PyVCAM,
    in order to make this code work, you need to install the PyVCAM package in the UUTrack.Controller.photometrics folder.
    and also need to install the MS visual studio and PVCAM SDK in the computer.

    .. sectionauthor:: Zhu Zhang <z.zhang@uu.nl>
"""
import numpy as np
from pint import UnitRegistry
import logging
import warnings
from typing import Tuple
from UUTrack.log import get_logger, log_to_file, log_to_screen
from pypylon import pylon
from UUTrack.Controller.devices.basler.basler_camera import BaslerCamera
import time
import sys,os
currentcwd = os.getcwd()
photometricsDriverPath = os.path.join(currentcwd, "UUTrack\Controller\devices\photometrics\src")
sys.path.append(photometricsDriverPath)
from pyvcam import pvc
from pyvcam.camera import Camera

from ._skeleton import cameraBase

ureg = UnitRegistry()
Q_ = ureg.Quantity
logger = get_logger(__name__)


class camera(cameraBase):
    MODE_CONTINUOUS = 1
    MODE_SINGLE_SHOT = 0
    MODE_EXTERNAL = 2

    def __init__(self, camera):
        super().__init__(camera)
        self.cam_num = camera  # Monitor ID
        # self.camera = BaslerCamera(camera)
        self.running = False
        self.mode = self.MODE_CONTINUOUS
        self.maxWidth = 0
        self.maxHeight = 0
        self.width = None
        self.height = None
        self.fps = 1
        # self.mode = None
        self.X = None
        self.Y = None
        self.friendly_name = None

    def initializeCamera(self):
        """ Initializes the communication with the camera. Get's the maximum and minimum width. It also forces
        the camera to work on Software Trigger.

        .. warning:: It may be useful to integrate other types of triggers in applications that need to
            synchronize with other hardware.

        """
        logger.debug('Initializing Photometrics BSI Camera')
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
        # self.setAcquisitionMode(self.MODE_SINGLE_SHOT)

    def triggerCamera(self):
        """Triggers the camera.
        """

        # if self.camera.IsGrabbing():
        #     logger.warning('Triggering an already grabbing camera')
        # else:
        if self.mode == self.MODE_CONTINUOUS:
            self.camera.start_live()
            #self.camera.start_live(exp_time=None, buffer_frame_count=16, stream_to_disk_path= 'C:\data\')
            #print("live mode")
    #         self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        elif self.mode == self.MODE_SINGLE_SHOT:
            print("Snap trigger mode")

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

    def setExposureUnit(self, expoUnit='ms'):
        """"set the exposure time unit in ms or us, defalt is in ms
        {'One Millisecond': 0, 'One Microsecond': 1}
        """
        if expoUnit == 'ms':
            self.camera.exp_res = 0
        elif expoUnit == 'us':
            self.camera.exp_res = 1
        else:
            print("set exposure failed, please choose the unit from ['ms' and 'us'].The exposure unit remain in %s " %self.checkExposureUnit())
        print("set the exposure unit to %s" % (self.checkExposureUnit()))
        return self.checkExposureUnit()

    def setExposure(self, exposure):  # done

        """ Set the exposure time of the camera in the current unit
        """
        self.camera.exp_time = int(exposure)
        self.exposure = exposure
        print("set the exposure time to %d in unit of %s" % (self.getExposure(), self.checkExposureUnit()))
        return self.getExposure()

    def getExposure(self):  # done
        """ Get the exposure time of the camera in the current unit
        """
        self.exposure = float(self.camera.exp_time)
        return self.exposure
    def getAmplifierSpeed(self):
        """ Get the speed of the camera ADC amplifier
            200MHz: 0
            100MHz: 1
        """
        if self.camera.speed_table_index == 0:
            self.ADCSpeed = '200MHz'
        elif self.camera.speed_table_index == 1:
            self.ADCSpeed = '100MHz'

        return self.ADCSpeed
    def setAmplifierSpeed(self, value):
        """ Get the speed of the camera ADC amplifier
            200MHz: 0
            100MHz: 1
        """
        if value in '200MHz':
            self.camera.speed_table_index = 0
        elif value in '100MHz':
            self.camera.speed_table_index = 1
        else: # default setting
            self.camera.speed_table_index = 1
        return self.getAmplifierSpeed()

    def getAmplifierGain(self):
        """ Get the gain of the camera ADC amplifier
        if ADC amplifier speed is 100 MHz
            HDR (16bit): 1
            CMS(12 bit): 2
        if ADC amplifier speed is 200 MHz
            Full well (11 bit):1
            Balanced (11 bit):2
            Sensitivity (11 bit) :3
        """
        if self.getAmplifierSpeed() == '200MHz':
            if self.camera.gain == 1:
                self.ADCGain = 'Fullwell(11bit)'
            elif self.camera.gain == 2:
                self.ADCGain = 'Balanced(11bit)'
            elif self.camera.gain == 3:
                self.ADCGain = 'Sensitivity(11bit)'
        elif self.getAmplifierSpeed() == '100MHz':
            if self.camera.gain == 1:
                self.ADCGain = 'HDR(16bit)'
            elif self.camera.gain == 2:
                self.ADCGain = 'CMS(12bit)'

        return self.ADCGain

    def setAmplifierGain(self, value):
        """ Get the gain of the camera ADC amplifier
        if ADC amplifier speed is 100 MHz
            HDR(16bit): 1
            CMS(12bit): 2
        if ADC amplifier speed is 200 MHz
            Full well (11 bit):1
            Balanced (11 bit):2
            Sensitivity (11 bit) :3
        """
        if value in 'HDR(16bit)':
            self.setAmplifierSpeed('100MHz')
            self.camera.gain = 1
        elif value in 'CMS(12bit)':
            self.setAmplifierSpeed('100MHz')
            self.camera.gain = 2

        elif value in 'Fullwell(11bit)':
            self.setAmplifierSpeed('200MHz')
            self.camera.gain = 1
        elif value in 'Balanced(11bit)':
            self.setAmplifierSpeed('200MHz')
            self.camera.gain = 2
        elif value in 'Sensitivity(11bit)':
            self.setAmplifierSpeed('200MHz')
            self.camera.gain = 3

        else: # default setting
            self.setAmplifierSpeed('100MHz')
            self.camera.gain = 1

        return self.getAmplifierGain()

    def get_adc(self):
        return self.camera.adc_offset

    def set_adc(self,value):

        self.camera.adc_offset = value

        return self.get_adc()
    def readCamera(self):
        # if not self.camera.IsGrabbing():
        #     raise WrongCameraState('You need to trigger the camera before reading from it')

        if self.mode == self.MODE_SINGLE_SHOT:
            grab = self.camera.get_frame(int(self.exposure),
                                         int(self.exposure + 200))  # (exp_time(int), timeout_ms(int))
            img = [grab.T]
            #
            print("single snap received, the shape is %s %s" % (img[0].shape[0], img[0].shape[1]))
            # self.camera.abort() # do not need the abort() the camera, the get_frame() function contains .finish() function

        # elif self.mode == self.MODE_CONTINUOUS:
        else:
            img = []
            self.num_buffers = 20
            logger.debug(f'{self.num_buffers} frames available')
            if self.num_buffers:
                img = [None] * self.num_buffers
                for i in range(self.num_buffers):
                    # try:
                    frame, self.fps, frame_count = self.camera.poll_frame(timeout_ms=int(self.exposure + 200),
                                        oldestFrame = True, copyData = True)
                    # frame = self.camera.get_frame(int(self.exposure),
                             #                    int(self.exposure + 50))
                    frame_array = frame['pixel_data'].T
                    img[i] = frame_array
                    #img[i] = frame.T
                    # print('Width:{} \tHeight:{} \tMaxValue:{} \tFrame Rate: {:.1f} \tFrame Count: {:.0f}\n'
                    #       .format(frame_array.shape[0], frame_array.shape[1], high, self.fps, frame_count))
        return img

    def getNframes(self, framesNum, exp_time, time_out):  # done
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
        ***** I think it could be a useful function to use in the future, but it is not used by me so far.
        """
        return self.camera.get_sequence(framesNum, exp_time, time_out)

    def set_bufferNum(self, bufferNUm):
        """ set the buffer numbers, in frames. but it is not used by me.
        """
        self.num_buffers = bufferNUm
        return self.num_buffers

    def setBinning(self, xbin, ybin):  # done
        """
        Sets the binning of the camera if supported. Has to check if binning in X/Y can be different or not, etc.
        :param xbin:
        :param ybin:
        :return:
        """
        currentBinng = self.getBinning()
        Nx, Ny = self.getSize()
        if xbin == 1 and ybin == 1:

            self.camera.binning = (1, 1)
            # self.maxWidth = self.CCDsize[0]
            # self.maxHeight = self.CCDsize[1]
            # if currentBinng == (2,2):
            #     self.setROI(currentSize[0], currentSize[1]*2)

        elif xbin == 2 and ybin == 2:
            self.camera.binning = (2, 2)
            # self.maxWidth = int(self.CCDsize[0] / 2)
            # self.maxHeight = int(self.CCDsize[1] / 2)
            print("CCD size after binning of 2 is %s %s " %(self.CCDsize[0], self.CCDsize[1]))
            # if currentBinng == (2,2):
            #     self.setROI(currentSize[0], currentSize[1]*2/2)
        else:
            self.camera.binning = (1, 1)

        return self.getBinning()
        # def set_ROI(self, X: Tuple[int, int], Y: Tuple[int, int]) -> Tuple[int, int]:

    def getBinning(self):  # done
        """
        get the binning of the camera if supported. Has to check if binning in X/Y can be different or not, etc.
        :return:
        """

        return self.camera.binning

    def setROI(self, X, Y):  # done
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
        print("call camera.setROI ")
        print(X)
        print(Y)
        logger.info(f'Updating ROI: (x, y, width, height) = ({x_pos}, {y_pos}, {width}, {height})')
        if x_pos + width > self.maxWidth:  # self.getSize() can update the ROI size based on the binning setting
            raise CameraException('ROI width bigger than current camera width of %d' % self.maxWidth)
        if y_pos + height > self.maxHeight:
            raise CameraException('ROI height bigger than camera height of %d' % self.maxHeight)

        # First clear the current ROI to avoid many ROIs, since the BSI allows to set 15 ROIs
        self.clear_ROI()
        self.camera.set_roi(x_pos, y_pos, width, height)
        logger.debug(f'Setting width to {width}')
        logger.debug(f'Setting X offset to {x_pos}')
        logger.debug(f'Setting Height to {height}')
        logger.debug(f'Setting Y offset to {y_pos}')
        self.X = (x_pos, x_pos + width)
        self.Y = (y_pos, y_pos + height)
        self.width = self.camera.shape(0)[0]
        self.height = self.camera.shape(0)[1]

        return self.getSize()

    def clear_ROI(self):  # done
        """ Resets the ROI to the maximum area of the camera.
        # thiw will return to the ROI to 2048 x 2048, and change the binning size to 1
        if the current binning sie is 2, and then we need to set the binning size back to 2
        """
        self.binning = self.getBinning()
        self.camera.reset_rois()
        self.setBinning(self.binning[0], self.binning[1])
        print("binging is %s %s"%(self.binning[0], self.binning[1]))
        return self.getSize()

    def getSize(self):  # done
        """Returns the size in pixels of the image being acquired. This is useful for checking the ROI settings.
        """
        return self.camera.shape(0)  # we only set 1 ROI, so ROI_index = 0

    def getSerialNumber(self):  # done
        """Returns the serial number of the camera.
        """
        # return self.camera.getModelInfo(self.cam_id)
        return self.camera.serial_no

    def GetCCDWidth(self):  # done
        """
        Returns
        The CCD width in pixels
        """
        # return self.camera.max_width
        return self.camera.sensor_size[0]

    def GetCCDHeight(self):  # done
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
        if self.fps == 1:
            fps = 1000/self.exposure/2
        elif self.fps > 1:
            fps = self.fps
        else:
            fps = 30
        # return 1 / self.camera.pix_time
        return fps

    def setMaxFPS(self, FPS):
        """
        Sets the maximum framerate for aquisition.
        :param FPS:
        :return:
        """
        return FPS
    def MaxFPS_T(self):
        """
        Get the maximum FPS which can theoretically get by a given exposure time
        """
        if self.checkExposureUnit() == 'ms':
            fpsMAX = 1000 / self.getExposure()
        elif self.checkExposureUnit() == 'us':
            fpsMAX = 1000 * 1000 / self.getExposure()
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
        if self.camera.fan_speed == 0:
            fanspeed = "HighSpeed"
        elif self.camera.fan_speed == 1:
            fanspeed = "MiddleSpeed"
        elif self.camera.fan_speed == 2:
            fanspeed = "LowSpeed"
        elif self.camera.fan_speed == 3:
            fanspeed = "WaterCooling"
        return fanspeed

    def getTemp(self):  # done
        """"get the sCMOS sensor temperature
        """
        return self.camera.temp

    def setTemp(self, temperature):
        """set the sensorr temperature
        """
        self.camera.temp_setpoint = temperature
        return self.camera.temp_setpoint
    def getADC_Offset(self):
        """" sReturns the camera's current ADC offset value.
        """
        return self.camera.adc_offset
    def getBit_depth(self):
        """" Returns the bit depth of pixel data for images collected with this camera. Bit depth cannot be changed directly;
        instead, users must select a desired speed table index value that has the desired bit depth.
        Note that a camera may have additional speed table entries for different readout ports.
        See Port and Speed Choices section inside the PVCAM User Manual for a visual representation of a speed table and to see
        which settings are controlled by which speed table index is currently selected.
        """
        return self.camera.bit_depth

    def getGain(self):
        """"Returns the current gain index for a camera. A ValueError will be raised if an invalid gain index is supplied to the setter.
        """
        return self.camera.gain

    def setGain(self, gainmode):
        """setthe current gain index for a camera. A ValueError will be raised if an invalid gain index is supplied to the setter.
        """
        self.camera.gain = gainmode
        return self.camera.gain

    def stopAcq(self):  # done
        # self.camera.stopAcquisition()
        logger.info('Stopping camera')

        self.camera.finish()


    def stopCamera(self):  # done
        """Stops the acquisition and closes the connection with the camera.
        """
        try:
            # Closing the camera
            self.camera.close()
            pvc.uninit_pvcam()
            return True
        except:
            # Monitor failed to close
            return False

if __name__ == "__main__":
    import time

    camera = camera("PM")
    camera.initializeCamera()
    print("Camera is %s" % camera.friendly_name)
    print("binning setting is %d %d" % (camera.getBinning()[0], camera.getBinning()[1]))
    camera.clear_ROI()
    a = camera.getSize()
    camera.setROI([1, 600], [11, 700])
    c = camera.getSize()
    fanspeed = camera.getFanSpeed()
    print("The Fan speed is %s" % fanspeed)
    temperature = camera.getTemp()
    print("Sensor temperature is %s" % temperature)
    new_set = camera.setTemp(-20)
    new_temp = camera.getTemp()
    print("Sensor new set is %s and temperature now is %s" % (new_set, new_temp))
    print('ROI size is %d %d' % (a[0], a[1]))
    print('New ROI size is %d %d' % (c[0], c[1]))
    print("max width is %d " % camera.GetCCDWidth())
    print("max Height is %d " % camera.GetCCDHeight())
    camera.setExposureUnit('ms')
    camera.setExposure(exposure=4)
    print("exposure time is %.1f in unit of %s" % (camera.getExposure(), camera.checkExposureUnit()))
    b = camera.setBinning(2, 2)

    camera.setROI([1, 100], [11, 700])
    print("check the binnin value %d" % camera.getBinning()[0])
    print("x binning size is %d" % (b[0]))
    print("x binning size is %d" % (b[1]))
    print("Original CCD size")
    print(camera.CCDsize)
    print("CCD size after binning")
    print(camera.camera.sensor_size)
    print("new ROI after Binning is %d x %d\n " % (camera.getSize()[0], camera.getSize()[1]))
    camera.setROI([1, 100], [11, 200])
    d = camera.getSize()
    print('New ROI size is %d %d' % (d[0], d[1]))
    print(camera.camera.shape(0))
    print('set binning to %d' % (camera.setBinning(1, 1)[0]))
    print("new roi is ")
    print(camera.camera.shape(0))

    print('serialnumber is %s' % (camera.getSerialNumber()))
    print('internal fps is %.1f' % camera.get_internal_fps())
    print("set max fps is %.1f" % camera.setMaxFPS(5))
    print('internal fps is %.1f' % camera.get_internal_fps())
    cmaeraspeedTable = camera.camera.port_speed_gain_table
    print(cmaeraspeedTable)
    print("exposure modes")
    print(camera.camera.exp_modes)
    print(camera.camera.exp_out_modes)

    print("bit depth is ---> %d" % camera.camera.bit_depth)
    print("Gain is ---> %d" % camera.camera.gain)
    print("Trigger table is --->%s" % camera.camera.trigger_table)
    print("ADC offset is --->%d" % camera.camera.adc_offset)
    print("exposure resolution index is  --->%s" % camera.camera.exp_res_index)

    print("exposure resolution value is  --->%s" % camera.camera.exp_res)
    print("exposure resolution unit is  --->%s" % camera.checkExposureUnit())
    exp_resolution_table = camera.camera.exp_resolutions
    print("exposure resolution table is  --->%s" % camera.camera.exp_resolutions)

    camera.setAcquisitionMode(camera.MODE_SINGLE_SHOT)
    d = camera.getAcquisitionMode()
    print('acquisition mode is %s' % d)
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
