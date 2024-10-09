## UUTrack folder of PDSM

Holds all the files and folders concerning the aquisition of data for the PDSM setup. Uses a heavily modified version of the UUTrack software.

## Folders and files

### Controller 
Holds the backend (I hope I use this term correctly) support for using various cameras with the UUTrack program.

### Model
Holds the code for saving data from the camera, correctly communicating with the camera and the code for remembering, using and loading the settings set in the UUTrack program.

### View
Holds the code for creating the GUI (guided user interface) of UUTrack and all its options. 

### DAQwavegeneration.py
Side program to use the DAQ as a wavegenerator. Usefull as a reference for writing new code with the DAQ, however not all code works.

### startMonitor.py 
Main starting point of the UUTrack program. Run it from a terminal with the correct python environment to start the program.
