"""
    UUTrack.View.Camera.specleWidget_time.py
    ===================================
    Creates a new window that shows the average deviation of pixel intensity in time
    by showing the average of normalized variation for a few frames of different pixels 
    at different starting times. 
    
    Should be used instead of the time version (this requires changing the import code in monitormain)
    
    .. sectionauthor:: Kevin Namink <k.w.namink@uu.nl>
"""


import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtGui


class SpecleWindow(QtGui.QMainWindow):
    """
    Simple window that relies on its parent for updating a 1-D plot.
    """

    def __init__(self, parent=None):
        super(SpecleWindow, self).__init__(parent=parent)
        self.cc = pg.PlotWidget()
        self.setCentralWidget(self.cc)
        self.parent = parent
        self.p = self.cc.plot()
        changing_label = QtGui.QLabel()
        font = changing_label.font()
        font.setPointSize(16)
        self.text = pg.TextItem(text='', color=(200, 200, 200), border='w', fill=(0, 0, 255, 100))
        self.text.setFont(font)
        self.cc.addItem(self.text)
        
        self.Npoints_x = 4
        self.Npoints_y = 4
        self.dataFrameLength = 60
        self.dataArray = np.zeros(self.dataFrameLength)
        self.dataLoc = 0
        
        self.lengthofplot = self.dataFrameLength
        self.sizeofplot = 20
        self.plotMatrix = np.full((self.sizeofplot, self.lengthofplot) , 1.)
        self.plotLoc = 0

        self.cc.setXRange(0, self.lengthofplot)
        self.cc.setYRange(0.9, 1.1)
        self.text.setPos(0, 1)
        self.text.setText("Blank image")
        self.plotted = np.full(self.sizeofplot, 1.)

    def addDataPoints(self, image):
        """ Update the data, called externally on every frame taken (when running)
        """
        xlen, ylen = image.shape  # Get shape of image
        self.dataLoc += 1  # Cycle trough data locations
        if self.dataLoc == self.dataFrameLength:
            self.dataLoc = 0
            
        if (xlen - 1 > self.Npoints_x) and (ylen - 1 > self.Npoints_y):
            newpointsvalue = np.mean(image[1::int(xlen/self.Npoints_x), 1::int(ylen/self.Npoints_y)])  # Get mean value of some points in image
        else:
            return
        
        self.dataArray[self.dataLoc] = newpointsvalue  # Save image value in data


    def getNextValues(self):
        # Add the variance of at most the center sizeofSROI squared pixels of the frame to the plot
        # Use binning for SROIbinning squared size bins
        if len(self.parent.tempimage) < 2:
            return 0, 0, 0  # Failsafe
        
        # 'Calculating' the variance and mean:
        var = np.std(self.dataArray)
        mean = self.dataArray[self.dataLoc]
        
        # Calculate next intensity variation array after ensuring not deviding by 0:
        if self.dataArray[self.dataLoc] == 0:
            self.dataArray[self.dataLoc] = 1
        values = np.abs(np.roll(self.dataArray, -self.dataLoc)/self.dataArray[self.dataLoc] - 1) + 1
            
        return var, mean, values

    def update(self):
        """ Updates the plot. It is called externally from the main window.
        """
        if not self.isVisible():
            return
        if self.parent is not None and self.parent.acquiring:
            if len(self.parent.tempimage) > 2:
                self.plotLoc += 1  # Cycle trough plot locations
                if self.plotLoc == self.sizeofplot:
                    self.plotLoc = 0
                    
                var, mean, measure = self.getNextValues()  # Fetch values
                self.plotMatrix[self.plotLoc,:] = measure[:]  # Update plotMatrix
                
                if isinstance(measure[0], (int, float)) and measure[0] == measure[0]:  # Test for non-numbers
                    self.plotted = np.mean(self.plotMatrix, axis=0)  # Update plot
                
                if self.plotted[-1] > 1.:
                    minimum, maximum = self.plotted.min(), self.plotted.max()
                    self.cc.setYRange(minimum - (maximum-minimum)/5, maximum + (maximum-minimum)/2)
                    self.text.setPos(0, maximum + (maximum-minimum)/2)
                n_maxvalued = (self.parent.tempimage.ravel() == 65535).sum()
                
                self.p.setData(self.plotted)
                self.text.setText('Plotting mean deviation per frame. Number max I.: %.3e\nCurrent variance:~%.3e     Current mean:~%.3e' % (n_maxvalued, var, mean))

            else:
                self.text.setPos(0, 1)
                self.text.setText("Blank image")
                self.cc.setYRange(0.9, 1.1)










