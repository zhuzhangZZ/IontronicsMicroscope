# instruments controlling fro iSCAT and total interal Scattering Microscopy #
instrumentation project based on the Pynta module, previously UUTrack packages


This program can be used for montoring the frames from the photometricsBSI camera and Basler camera, and synchronize the potentiostat with cameras.
Frames data and CV data can be accumulated in a queue for saving while acquiring or saving retroactively.



## Software for monitoring a camera. ##
The program follows the Model-View-Controller design structure. This allows a rapid exchange of different parts of the code.
## Software for monitoring a potentiostat. ##
The program is dedicated to make the potentiostat work together with the camera. 

### Structure of the folders: ###
UUTrak: Main folder. Important executables should be placed here.

* _Controller_ : Houses the files related to periferals, such as python wrappers for cameras. They are organized inside of folders according to the brand. The idea is to copy/paste wrappers already available, without worrying for specific implementations.

* _Model_: Houses the intermediate steps between model and View. It handles the conditioning of data before being presented to the user. A model has to be defined for each different camera and for each different experiment. The skeleton should house all the used functions exposed to the user, in this way, if an implementation has the same functions with the same outputs, nothing will break downstream. Each class here inherits directly from the Controller device; this allows to access lower-level functionality without explicitly importing the Controller modules.

* _View_: Houses everything related to visualization of data. View should communicate only through models to devices and should get the input from the user. Acquisition tasks should be performed in a different thread, in order not to block the GUI. A timer updates the GUI at constant intervals, while the acquisition can happen at a different rate.


###  code

The code is a python package created as a GUI (Guided User Interface) for the setup in ITPDYS in PAris for the setup with iSCAT and TIRS. 



### Generated Data  

When finishing a measurement with UUTrack you have a number of \texttt{\allowbreak "(..).hdf5"} and \texttt{\allowbreak "(..)\_m\#.npy"\allowbreak } files, where \texttt{\allowbreak (..)} is what you called the measurement. 
These files you can analyze using the file \texttt{\allowbreak "PDSM/Analysis/\allowbreak data\_processing/\allowbreak new\_analysis\_file.py"}. 
This analysis file imports the data, identifies some properties of the data, helps you select a particle of interest, applies drift correction, averages over cycles and plots various possibly interesting sets of data. 
Among the final plots are intensity versus time and intensity versus applied potential.

This analysis file starts with a part containing some settings that speak for themselves and then there are various blocks doing the steps for analysis. 
Each block starts with a very short explanation what it does and how to use it.
While executing a block it will show some preliminary results, that sometimes help you find the best settings. 



### dependencies

Python 3.6 with the following packages:

```
hardpatato
pyVCAM
alabaster==0.7.10
argh==0.26.2
Babel==2.4.0
cffi==1.11.5
colorama==0.3.7
cycler==0.10.0
docutils==0.13.1
h5py==2.8.0
imagesize==1.1.0
Jinja2==2.9.5
kiwisolver==1.0.1
Lantz==0.3
livereload==2.6.0
MarkupSafe==1.0
matplotlib==3.0.2
numpy==1.15.4
olefile==0.46
pathtools==0.1.2
Pillow==5.3.0
Pint==0.8.1
pip==18.1
port-for==0.4
psutil==5.4.8
pycparser==2.19
PyDAQmx==1.4.2
Pygments==2.2.0
PyOpenGL==3.1.0
pyparsing==2.3.0
PyQt4==4.11.4
pyqtgraph==0.10.0
python-dateutil==2.7.5
pytz==2018.7
PyVISA==1.9.1
PyYAML==3.13
requests==2.13.0
scipy==1.1.0
setuptools==40.6.2
six==1.11.0
snowballstemmer==1.2.1
stringparser==0.4.1
tornado==4.4.3
watchdog==0.9.0
wheel==0.32.3
```

Like mentioned before, when using a fresh install it is quite possible there are difficult to install packages. 


### /PDSMGUI

This */Config* folder holds the used configuration files. 

There are various text and code files that speak for themselves. 
The *startProgram.py* file is called with python to start the software.

The folder */UUTrack* holds this setups highly edited version of UUTrack and is documented in the coming subsections.


### /UUTrack
*startMonitor.py* is important as code. 
It is the code that actually starts the software and when changing cameras it is quite possible that this code needs to be edited apart from the configuration file because it is slightly hard coded.
*DAQwavegeneration.py* is a seperate program to use the DAQ as a wavegenerator. Usefull as a reference for writing new code with the DAQ, however not all code inside it work.


### /UUTrack/Controller/devices
In each folder here there is the basic initial interface for using different camera's with the python program. 
The Hamamatsu is definitely working, the Basler is probably working and the rest has not been checked.

### /UUTrack/Model
Holds the code for saving data from the camera, correctly communicating with the camera and the code for remembering, using and loading the settings set in the UUTrack program.

### /UUTrack/Model/Cameras
This folder holds the python files which 'represent' the possible cameras the program can use. 
They should work if the required files in the devices folder are working correctly. 
The name of these files is used when configuring what camera to use if you want to change the used camera in the code, some cameras will not be possible to change to using the config file without changing the code in various places. 


### /UUTrack/View
The *hdfloader.py* can be very useful when taking a look at the created data and the */Monitor* folder holds a lot of code which will be the next section.


### /UUTrack/View/Monitor
This is the big one. 
With all the behind the scenes stuff out of the way, the code here will directly appear on screen when running the files. 
I will go trough each file quickly because it is reasonably commented.

##### Icons
A folder with all icons used in *monitorMain.py*.

#### LocateParticle.py
Currently unused code, kept here because it might be useful in the future.
The LocateParticle class contains necessary methods for localizing the centroid of a particle and following its track. 

#### NIcontrol.py
File with a class used to communicate with the NI DAQ correctly for this setup. 
It sends the commands to start and stop to measure all required channels with the camera as a trigger and to first flash a LED to synchronize the data. 
It also holds the code for generating waves with varying offset with the DAQ from the UUTrack GUI.

#### cameraViewer.py
Currently unused code, it is supposed to start an extra window which only shows the current camera image. 
Kept here because it might be useful in the future.

#### clearQueueThread.py
This file holds a class that clears the queue from the "QtCore.QThread", I think it is not used. 
I expect it to have been useful when debugging and might be in the future.

#### configWidget.py
This file creates the "Parameters" tile/widget in the main window. 
This widget can be used to view and change the settings that have been set in the config file supplied when starting the program.

#### contrastViewer.py
Creates a window that displays a 1D plot of a cross cut on the main window.
It is currently unused. 

#### crossCut.py
Plots total intensity value of line trough image while taking background corrections into account.
Its functional but not useful in practice.

#### mainImageManipulation.py
Holds a class that manipulates the mean image seen in the GUI. 
Is required to correctly use binning and the various background correction methods for the GUI. 
Any change it makes to the image shown in the GUI is NOT saved when recording data.

#### messageWidget.py
Is used to create "Messages" tile/widget in the main window. 
It shows a permanent message for every important action done by the user in a log style and it shows the status of some parameters from the camera and measurement.

#### monitorMain.py
This is a very large file which is responsible for creating the main window with all its widgets. 
It has all the code necessary to use the various widgets from the different python files in this folder and to make this as easy to use as possible there is also a lot of code concerning shortcuts and menu bars. 
Because of the length of the *monitorMain.py* file you might think it does a lot of calculations as well, but this is almost all done in the different imported files it calls from. 


#### monitorMainWidget.py
This file holds the code responsible for the large preview of the camera image in the main window. 
This window shows what the camera can currently see and it is possible to change the range of interest in this widget. 
There is also the possibility to plot the intensity at a certain location in the plot by pressing alt while hovering over the image, this is a feature that is not recommended for use but can be expanded on in the future.

#### popOut.py
Creates a pop-out window class that can be used to show information. 
Is currently only used to create a pop-out that shows all the keyboard shortcuts available.

#### resources.py
This is "resource object code". It is probably important.

#### specialTaskTrack.py
Similar to the "UUTrack.View.Camera.workerThread", the special task worker is designed for running in a separate thread a task other than just acquiring from the camera.
For example, one can use this to activate some feedback loop.
This is unused code but has been used as a reference for how to write other code.
When implementing some sort of particle tracking this might be a useful file, so it is kept.

#### specleWidget\_space.py
Creates a new window that shows the (spatial) variance of the speckle pattern in the center of the current ROI in the image over time and some parameters like the amount of oversaturated pixels in the image.
Interchangeable with the *specleWidget\_time.py* file, which is thought to be more useful.

#### specleWidget\_time.py
Creates a new window that shows the average deviation of pixel intensity in time by showing the average of normalized variation for a few frames of different pixels at different starting times. 
It also shows some parameters like the amount of oversaturated pixels in the image.
Interchangeable with the \texttt{specleWidget\_space.py} file, but this one is probably more useful.

#### trajectoryWidget.py
Creates a widget in the main window to plot the intensity of a pixel selected by holding alt and hovering over the preview image in the GUI. 
It is not very useful as it is right now but for potential future uses it is kept. 

#### waterfallWidget.py
A not very useful, but functional, widget that plots a waterfall plot. 
Perhaps can be changed to something else in the future.

#### workerThread.py
Contains a thread class that acquires continuously data until a variable is changed. 
This enables to acquire at any frame rate without freezing the GUI or overloading it with data being acquired too fast.



### Workplan and Issue tracker ###

check [WORKPLAN](WORKPLAN.txt) file (please link here)
