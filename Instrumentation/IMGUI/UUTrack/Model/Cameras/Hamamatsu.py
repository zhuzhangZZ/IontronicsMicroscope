"""
    UUTrack.Model.Cameras.Hamamatsu.py
    ==================================

    Model class for controlling Hamamatsu cameras via de DCAM-API. At the time of writing this class,
    little documentation on the DCAM-API was available. Hamamatsu has a different time schedule regardin support of
    their own API. However, Zhuang's lab Github repository had a python driver for the Orca camera and with a bit of
    tinkering things worked out.

    DCAM-API relies mostly on setting parameters into the camera. The correct data type of each parameter is not well
    documented; however it is possible to print all the available properties and work from there. The properties are
    stored in a filed named params.txt next to the :mod:`Hamamatsu Driver
    <UUTrack.Controller.devices.hamamatsu.hamamatsu_camera>`

    .. note:: When setting the ROI, Hamamatsu only allows to set multiples of 4 for every setting (X,Y and vsize,
        hsize). This is checked in the function. Changing the ROI cannot be done directly, one first needs to disable it
        and then re-enable.


    .. sectionauthor:: Aquiles Carattino <aquiles@aquicarattino.com>
"""
import numpy as np

from UUTrack.Controller.devices.hamamatsu.hamamatsu_camera import HamamatsuCamera
from ._skeleton import cameraBase


class camera(cameraBase):
    MODE_CONTINUOUS = 1
    MODE_SINGLE_SHOT = 0
    MODE_EXTERNAL = 2

    def __init__(self,camera):
        self.cam_id = camera # Monitor ID
        self.camera = HamamatsuCamera(camera)
        self.running = False
        self.mode = self.MODE_SINGLE_SHOT

    def initializeCamera(self):
        """ Initializes the camera.

        :return:
        """

        self.camera.initCamera()
        self.maxWidth = self.GetCCDWidth()
        self.maxHeight = self.GetCCDHeight()
        #This is important to not have shufled patches of the CCD.
        #Have to check documentation!!
        self.camera.setPropertyValue("readout_speed", 1)
        self.camera.setPropertyValue("defect_correct_mode", 1)

    def triggerCamera(self):
        """Triggers the camera.
        """
        if self.getAcquisitionMode() == self.MODE_CONTINUOUS:
            self.camera.startAcquisition()
        elif self.getAcquisitionMode() == self.MODE_SINGLE_SHOT:
            self.camera.startAcquisition()
            self.camera.stopAcquisition()

    def setAcquisitionMode(self, mode):
        """
        Set the readout mode of the camera: Single or continuous.
        Parameters
        mode : int
        One of self.MODE_CONTINUOUS, self.MODE_SINGLE_SHOT
        """
        self.mode = mode
        if mode == self.MODE_CONTINUOUS:
            #self.camera.setPropertyValue("trigger_source", 1)
            self.camera.settrigger(1)
            self.camera.setmode(self.camera.CAPTUREMODE_SEQUENCE)
        elif mode == self.MODE_SINGLE_SHOT:
            #self.camera.setPropertyValue("trigger_source", 3)
            self.camera.settrigger(1)
            self.camera.setmode(self.camera.CAPTUREMODE_SNAP)
        elif mode == self.MODE_EXTERNAL:
            #self.camera.setPropertyValue("trigger_source", 2)
            self.camera.settrigger(2)
        return self.getAcquisitionMode()

    def getAcquisitionMode(self):
        """Returns the acquisition mode, either continuous or single shot.
        """
        return self.mode

    def acquisitionReady(self):
        """Checks if the acquisition in the camera is over.
        """
        return True

    def setExposure(self,exposure):
        """
        Sets the exposure of the camera.
        """
        self.camera.setPropertyValue("exposure_time",exposure/1000)
        return self.getExposure()

    def getExposure(self):
        """
        Gets the exposure time of the camera.
        """
        return self.camera.getPropertyValue("exposure_time")[0]*1000

    def readCamera(self):
        """
        Reads the camera
        """
        [frames, dims] = self.camera.getFrames()
        img = []
        for f in frames:
            d = f.getData()
            d = np.reshape(d, (dims[1], dims[0]))
            d = d.T
            img.append(d)
#        img = frames[-1].getData()
#        img = np.reshape(img,(dims[0],dims[1]))
        return img

    def setBinning(self,xbin,ybin):
        """
        Sets the binning of the camera if supported. Has to check if binning in X/Y can be different or not, etc.

        :param xbin:
        :param ybin:
        :return:
        """
        if xbin == 1 and ybin == 1:
            self.camera.setPropertyValue("binning", 1)
        elif xbin == 2 and ybin == 2:
            self.camera.setPropertyValue("binning", 2)
        elif xbin == 4 and ybin == 4:
            self.camera.setPropertyValue("binning", 4)
            
        return self.camera.getPropertyValue("binning")
        
    
    
    def setMaxFPS(self,FPS):
        """
        Sets the maximum framerate for aquisition.  DOES NOT WORK
        
        :param FPS:
        :return:
        """
        if FPS >= 200:
            self.camera.setPropertyValue("master_pulse_interval", 1./200)
        else:
            self.camera.setPropertyValue("master_pulse_interval", 1./FPS)
            
        print("dunnit")
        return self.camera.getPropertyValue("master_pulse_interval")
        
    
    
    def setROI(self,X,Y):
        """
        Sets up the ROI. Not all cameras are 0-indexed, so this is an important
        place to define the proper ROI.
        X -- array type with the coordinates for the ROI X[0], X[1]
        Y -- array type with the coordinates for the ROI Y[0], Y[1]
        """
        # First needs to go full frame, if not, throws an error of subframe not valid
        self.camera.setPropertyValue("subarray_vpos", 0)
        self.camera.setPropertyValue("subarray_hpos", 0)
        self.camera.setPropertyValue("subarray_vsize", self.camera.max_height)
        self.camera.setPropertyValue("subarray_hsize", self.camera.max_width)
        self.camera.setSubArrayMode()

        X-=1
        Y-=1
        # Because of how Orca Flash 4 works, all the ROI parameters have to be multiple of 4.
        hsize = int(abs(X[0]-X[1])/4)*4
        hpos = int(X[0]/4)*4
        vsize = int(abs(Y[0]-Y[1])/4)*4
        vpos = int(Y[0]/4)*4
        self.camera.setPropertyValue("subarray_vpos", vpos)
        self.camera.setPropertyValue("subarray_hpos", hpos)
        self.camera.setPropertyValue("subarray_vsize", vsize)
        self.camera.setPropertyValue("subarray_hsize", hsize)
        self.camera.setSubArrayMode()
        return self.getSize()

    def getSize(self):
        """Returns the size in pixels of the image being acquired. This is useful for checking the ROI settings.
        """
        X = self.camera.getPropertyValue("subarray_hsize")
        Y = self.camera.getPropertyValue("subarray_vsize")
        return X[0], Y[0]

    def getSerialNumber(self):
        """Returns the serial number of the camera.
        """
        return self.camera.getModelInfo(self.cam_id)

    def GetCCDWidth(self):
        """
        Returns
        The CCD width in pixels

        """
        return self.camera.max_width

    def GetCCDHeight(self):
        """
        Returns
        The CCD height in pixels

        """
        return self.camera.max_height

    def stopAcq(self):
        self.camera.stopAcquisition()

    def stopCamera(self):
        """Stops the acquisition and closes the connection with the camera.
        """
        try:
            #Closing the camera
            self.camera.stopAcquisition()
            self.camera.shutdown()
            return True
        except:
            #Monitor failed to close
            return False
