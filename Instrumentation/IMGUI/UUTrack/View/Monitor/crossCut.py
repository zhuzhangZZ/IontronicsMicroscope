"""
    UUTrack.View.Camera.crossCut.py
    ===================================
    Plots total intensity value of line trough image. (Takes background into account)
    
    .. sectionauthor:: Kevin Namink
    .. sectionauthor:: Zhu Zhang <z.zhang@uu.nl>
    Based on code by:
    .. sectionauthor:: Aquiles Carattino <aquiles@aquicarattino.com>

"""

import pyqtgraph as pg
import numpy as np
import copy
from pyqtgraph.Qt import QtGui
import time


class crossCutWindow_0(QtGui.QMainWindow):
    """
    Simple window that relies on its parent for updating a 1-D plot.
    """
    def __init__(self, parent=None):
        super(crossCutWindow, self).__init__(parent=parent)
        self.cc = pg.PlotWidget()
        self.setCentralWidget(self.cc)
        self.parent = parent
        y = np.random.random(100)
        self.p = self.cc.plot()
        changingLabel = QtGui.QLabel()
        font = changingLabel.font()
        font.setPointSize(16)
        self.text = pg.TextItem(text='', color=(200, 200, 200), border='w', fill=(0, 0, 255, 100))
        self.text.setFont(font)
        self.cc.addItem(self.text)
        self.cc.setRange(xRange=(0,100), yRange=(-20,500))
        self.x_data = list(np.arange(0,400) - 399)
        self.img_avg = list(np.ones(400)*100)
        print("crossCut initialized Done")

        """ Updates the 1-D plot. It is called externally from the main window.
        """
        if self.parent != None:
            if len(self.parent.tempimage) >= 2:
                # time_start = time.time()
                # print(time_start)
                len(self.parent.tempimage)
                s = self.parent.camWidget.crossCut.value()
                (w,h) = np.shape(self.parent.tempimage)
                self.cc.setXRange(0,w)
                if s<h:

                    # self.time_now = time.time() - time_start
                    d = copy.copy(self.parent.tempimage[:, s])
                    # self.img_avg_now = np.mean(self.parent.tempimage)
                    # self.x_data = self.x_data[1:]  # Remove the first y element.
                    # self.x_data.append(self.time_now)  # Add a new value 1 higher than the last.
                    #
                    # self.img_avg = self.img_avg[1:]  # Remove the first
                    # self.img_avg.append(self.img_avg_now)  # Add a new random value.

                    if (self.parent.mainImageManip.bgON):
                        bg = self.parent.mainImageManip.bgimage[:, s]
                        d = d - bg

                    self.p.setData(d)
                    # self.p.setData(self.x_data, self.img_avg)
                    if np.mean(d) > 0:
                        self.text.setText('Line %d\t, Average: %d\t Max: %d\t' %(s, np.mean(d), np.max(d)))
            else:
            # except:
                self.text.setText("Blank image")


import pyqtgraph as pg
import numpy as np
import copy
from pyqtgraph.Qt import QtGui


class crossCutWindow(QtGui.QMainWindow):
    """
    Simple window that relies on its parent for updating a 1-D plot.
    """
    def __init__(self, parent=None):
        super(crossCutWindow, self).__init__(parent=parent)
        self.cc = pg.PlotWidget()
        self.setCentralWidget(self.cc)
        self.parent = parent
        self.p = self.cc.plot()
        changingLabel = QtGui.QLabel()
        font = changingLabel.font()
        font.setPointSize(16)
        self.text = pg.TextItem(text='', color=(200, 200, 200), border='w', fill=(0, 0, 255, 100))
        self.text.setFont(font)
        self.cc.addItem(self.text)
        self.cc.setRange(xRange=(0,100), yRange=(-20,500))
        self.x_data = list(np.arange(0, 400)-399)
        if len(self.parent.tempimage) >= 2:
            self.img_avg = list(np.ones(400) * np.average(self.parent.tempimage))
        else:
            self.img_avg = list(np.ones(400) * 1)
        self.cc.setLabel('bottom', text="Time(s)")
        self.cc.setLabel('left', text="Intensity")
        self.time_previous = time.time()
    def update(self):
        """ Updates the 1-D plot. It is called externally from the main window.
        """
        if self.parent != None and self.parent.crossCutUpdate is True:
            if len(self.parent.tempimage) >= 2:

                s = self.parent.camWidget.crossCut.value()
                (w,h) = np.shape(self.parent.tempimage)
                # self.cc.setXRange(0,w)

                if s<h:
                    self.time_now = time.time() - self.time_previous
                    d = copy.copy(self.parent.tempimage[:, s])

                    self.img_avg_now = np.mean(self.parent.tempimage)
                    self.x_data = self.x_data[1:]  # Remove the first y element.
                    self.x_data.append(self.time_now)  # Add a new value 1 higher than the last.
                    self.img_avg = self.img_avg[1:]  # Remove the first
                    self.img_avg.append(self.img_avg_now)  # Add a new random value.
                    if (self.parent.mainImageManip.bgON):
                        bg = self.parent.mainImageManip.bgimage[:, s]
                        d = d - bg
                    # self.p.setData(d)
                    self.p.setData(self.x_data, self.img_avg)
                    self.cc.setXRange(self.time_now - 10, self.time_now + 0.1)
                    self.cc.setYRange(min(self.img_avg)* 0.98, max(self.img_avg)*1.02)
                    if np.mean(d) > 0:
                        self.text.setText('Line %d\t, Average: %d\t Max: %d\t' %(s, np.mean(d), np.max(d)))
            else:
                self.text.setText("Blank image")

    def close_window(self):
        self.close()
if __name__ == '__main__':
    app = QtGui.QApplication([])
    win = crossCutWindow()
    x = np.random.normal(size=100)
    y = np.random.normal(size=100)
    win.cc.plot(x,y)
    win.show()
    app.instance().exec_()
