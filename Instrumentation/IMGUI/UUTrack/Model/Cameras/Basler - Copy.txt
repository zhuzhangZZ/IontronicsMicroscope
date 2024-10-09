"""
    UUTrack.Model.Cameras.Basler.py
    ==================================

    Model class for controlling Basler cameras via de Pylon API. 


    .. sectionauthor:: Zhu Zhang <z.zhang@uu.nl>
"""
import os
import numpy as np
from pint import UnitRegistry
import logging
import warnings
from typing import Tuple
from UUTrack.log import get_logger, log_to_file, log_to_screen
from pypylon import pylon
from UUTrack.Controller.devices.basler.basler_camera import BaslerCamera
import time
from ._skeleton import cameraBase

ureg = UnitRegistry()
Q_ = ureg.Quantity
logger = get_logger(__name__)
class camera(cameraBase):
    MODE_CONTINUOUS = 1
    MODE_SINGLE_SHOT = 0
    MODE_EXTERNAL = 2

    def __init__(self,camera):
        super().__init__(camera)
        # self.cam_id = camera # Monitor ID
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
        self.friendly_name = None
        print('start Basler camera and print dir %s'%os.getcwd())

    def initializeCamera(self):
        """ Initializes the communication with the camera. Get's the maximum and minimum width. It also forces
        the camera to work on Software Trigger.

        .. warning:: It may be useful to integrate other types of triggers in applications that need to
            synchronize with other hardware.

        """
        logger.debug('Initializing Basler Camera')
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if len(devices) == 0:
            raise CameraNotFound('No camera found')

        for device in devices:
            if self.cam_num in device.GetFriendlyName():
                self.camera = pylon.InstantCamera()
                self.camera.Attach(tl_factory.CreateDevice(device))
                self.camera.Open()
                self.friendly_name = device.GetFriendlyName()
                self.serialNumber = device.GetSerialNumber()

        if not self.camera:
            msg = f'{self.cam_num} not found. Please check your config file and cameras connected'
            logger.error(msg)
            raise CameraNotFound(msg)
        self.modelname = self.camera.GetDeviceInfo().GetModelName()
        logger.info(f'Loaded camera {self.modelname}')

        self.camera.ExposureAuto.SetValue('Off')
        self.maxWidth = self.camera.WidthMax.GetValue()
        self.maxHeight = self.camera.HeightMax.GetValue()
        offsetX = self.camera.OffsetX.GetValue()
        offsetY = self.camera.OffsetY.GetValue()
        width = self.camera.Width.GetValue()
        height = self.camera.Height.GetValue()
        self.X = (offsetX, offsetX + width)
        self.Y = (offsetY, offsetY + height)

        self.camera.RegisterConfiguration(pylon.SoftwareTriggerConfiguration(), pylon.RegistrationMode_ReplaceAll,
                                          pylon.Cleanup_Delete)
        self.setAcquisitionMode(self.MODE_SINGLE_SHOT)

    def triggerCamera(self):
        """Triggers the camera.
        """

        if self.camera.IsGrabbing():
            logger.warning('Triggering an already grabbing camera')
        else:
            if self.mode == self.MODE_CONTINUOUS:
                self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            elif self.mode == self.MODE_SINGLE_SHOT:
                self.camera.StartGrabbing(1)
        self.camera.ExecuteSoftwareTrigger()
    def setAcquisitionMode(self, mode):
        """
        Set the readout mode of the camera: Single or continuous.
        Parameters
        mode : int
        One of self.MODE_CONTINUOUS, self.MODE_SINGLE_SHOT
        """
        logger.info(f'Setting acquisition mode to {mode}')
        if mode == self.MODE_CONTINUOUS:
            logger.debug(f'Setting buffer to {self.camera.MaxNumBuffer.Value}')
            self.camera.OutputQueueSize = self.camera.MaxNumBuffer.Value
            self.camera.AcquisitionMode.SetValue('Continuous')
            self.mode = mode
        elif mode == self.MODE_SINGLE_SHOT:
            self.camera.AcquisitionMode.SetValue('SingleFrame')
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

    def setExposure(self, exposure):

        self.camera.ExposureTime.SetValue(exposure*1000)
        self.exposure = exposure
        return self.getExposure()

    def getExposure(self) :
        self.exposure = float(self.camera.ExposureTime.GetValue()/1000)
        return self.exposure

    def get_exposure_info(self):
        self.expo_unit = self.camera.ExposureTime.GetUnit()
        self.expo_min = self.camera.ExposureTime.Getmin()
        self.expo_max = self.camera.ExposureTime.GetMax()
    def readCamera(self):
        if not self.camera.IsGrabbing():
            raise WrongCameraState('You need to trigger the camera before reading from it')

        if self.mode == self.MODE_SINGLE_SHOT:
            grab = self.camera.RetrieveResult(int(self.exposure*1000) + 100*1000, pylon.TimeoutHandling_Return)
            img = [grab.Array]
            grab.Release()
            self.camera.StopGrabbing()
        else:
            img = []
            num_buffers = self.camera.NumReadyBuffers.Value
            logger.debug(f'{self.camera.NumQueuedBuffers.Value} frames available')
            if num_buffers:
                img = [None] * num_buffers
                for i in range(num_buffers):
                    grab = self.camera.RetrieveResult(int(self.exposure*1000) + 100*1000, pylon.TimeoutHandling_Return)
                    if grab:
                        img[i] = grab.Array
                        grab.Release()
        return [i.T for i in img]  # Transpose to have the correct size

    def set_bufferNum(self, bufferNUm):
        # self.camera.MaxNumBuffer = int(bufferNUm)
        self.camera.MaxNumBuffer.SetValue(int(bufferNUm))
        return self.camera.MaxNumBuffer.GetValue()

    def setBinning(self, xbin, ybin):
        """
        Sets the binning of the camera if supported. Has to check if binning in X/Y can be different or not, etc.
        :param xbin:
        :param ybin:
        :return:
        """
        if xbin == 1 and ybin == 1:
            self.camera.BinningHorizontal.SetValue(1)
            self.camera.BinningVertical.SetValue(1)
        elif xbin == 1 and ybin == 2:
            self.camera.BinningHorizontal.SetValue(1)
            self.camera.BinningVertical.SetValue(2)
        elif xbin == 2 and ybin == 1:
            self.camera.BinningHorizontal.SetValue(2)
            self.camera.BinningVertical.SetValue(1)
        elif xbin == 2 and ybin == 2:
            self.camera.BinningHorizontal.SetValue(2)
            self.camera.BinningVertical.SetValue(2)
        elif xbin == 4 and ybin == 4:
            self.camera.BinningHorizontal.SetValue(4)
            self.camera.BinningVertical.SetValue(4)

        return self.camera.BinningHorizontal.GetValue(), self.camera.BinningVertical.GetValue()
    # def set_ROI(self, X: Tuple[int, int], Y: Tuple[int, int]) -> Tuple[int, int]:
    def setROI(self, X, Y):
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
        if x_pos + width > self.maxWidth:
            raise CameraException('ROI width bigger than camera area')
        if y_pos + height > self.maxHeight:
            raise CameraException('ROI height bigger than camera area')

        # First set offset to minimum, to avoid problems when going to a bigger size
        self.clear_ROI()
        logger.debug(f'Setting width to {width}')
        self.camera.Width.SetValue(width)
        logger.debug(f'Setting X offset to {x_pos}')
        self.camera.OffsetX.SetValue(x_pos)
        logger.debug(f'Setting Height to {height}')
        self.camera.Height.SetValue(height)
        logger.debug(f'Setting Y offset to {y_pos}')
        self.camera.OffsetY.SetValue(y_pos)
        self.X = (x_pos, x_pos + width)
        self.Y = (y_pos, y_pos + width)
        self.width = self.camera.Width.GetValue()
        self.height = self.camera.Height.GetValue()
        return self.getSize()

    def clear_ROI(self):
        """ Resets the ROI to the maximum area of the camera"""
        self.camera.OffsetX.SetValue(self.camera.OffsetX.Min)
        self.camera.OffsetY.SetValue(self.camera.OffsetY.Min)
        self.camera.Width.SetValue(self.camera.WidthMax.GetValue())
        self.camera.Height.SetValue(self.camera.HeightMax.GetValue())
    def getSize(self):
        """Returns the size in pixels of the image being acquired. This is useful for checking the ROI settings.
        """
        return self.camera.Width.GetValue(), self.camera.Height.GetValue()
    def getSerialNumber(self):
        """Returns the serial number of the camera.
        """
        # return self.camera.getModelInfo(self.cam_id)
        return self.serialNumber

    def GetCCDWidth(self):
        """
        Returns
        The CCD width in pixels
        """
        warnings.warn("This method will be removed in a future release. Use cls.max_width instead", DeprecationWarning)
        # return self.camera.max_width
        return self.maxWidth

    def GetCCDHeight(self):
        """
        Returns
        The CCD height in pixels
        """
        # return self.camera.max_height
        warnings.warn("This method will be removed in a future release. Use cls.max_height instead", DeprecationWarning)
        return self.maxHeight
    def get_internal_fps(self):
        """
        Returns
        The CCD realtime frame rate
        """
        return self.camera.ResultingFrameRate.GetValue()

    def setMaxFPS(self, FPS):
        """
        Sets the maximum framerate for aquisition.  DOES NOT WORK

        :param FPS:
        :return:
        """
        if FPS <= self.camera.ResultingFrameRate.GetValue():
            self.camera.AcquisitionFrameRate.SetValue(FPS)
        else:
            self.camera.AcquisitionFrameRate.SetValue(self.camera.ResultingFrameRate.GetValue())

        print("dunnit")
        return self.camera.ResultingFrameRate.GetValue()
    def stopAcq(self):
        # self.camera.stopAcquisition()
        logger.info('Stopping camera')
        self.camera.StopGrabbing()
        self.camera.AcquisitionStop.Execute()
    def stopCamera(self):
        """Stops the acquisition and closes the connection with the camera.
        """
        try:
            #Closing the camera
            # self.camera.stopAcquisition()
            self.camera.StopGrabbing()
            time.sleep(0.5)
            self.camera.AcquisitionStop.Execute()
            # self.camera.shutdown()
            self.camera.Close()
            return True
        except:
            #Monitor failed to close
            return False


if __name__ == "__main__":
    import time

    camera = camera("puA")
    camera.initializeCamera()
    print("Camera is %s" % camera.friendly_name)

    camera.clear_ROI()
    a = camera.getSize()
    camera.setROI([1, 111], [11, 100])
    c = camera.getSize()

    print('ROI size is %d %d' % (a[0], a[1]))
    print('New ROI size is %d %d' % (c[0], c[1]))
    print("max width is %d " % camera.GetCCDWidth())
    print("max Height is %d " % camera.GetCCDHeight())
    camera.setExposure(exposure=2.2)
    print("exposure time is %.1f" % (camera.getExposure()))
    b = camera.setBinning(1, 1)
    print("x binning size is %d" % (b[0]))
    print("x binning size is %d" % (b[1]))
    print('serialnumber is %s' % (camera.getSerialNumber()))
    print('internal fps is %.1f' % camera.get_internal_fps())
    print("set max fps is %.1f" % camera.setMaxFPS(5))
    print('internal fps is %.1f' % camera.get_internal_fps())

    camera.setAcquisitionMode(camera.MODE_SINGLE_SHOT)
    d = camera.getAcquisitionMode()
    print('acquisition mode is %s' % d)
    camera.triggerCamera()
    shot = camera.readCamera()
    print("shot image length is %d" % len(shot))

    camera.setAcquisitionMode(camera.MODE_CONTINUOUS)
    e = camera.getAcquisitionMode()
    print('acquisition mode is %s' % e)
    camera.triggerCamera()
    time.sleep(2)
    frames = camera.readCamera()
    print(len(frames))
    camera.stopAcq()
    camera.stopCamera()
