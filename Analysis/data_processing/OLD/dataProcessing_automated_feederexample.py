# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 09:50:16 2019

@author: kevin
"""

import dataProcessing_automated as doData

#%%
# Process 1 movie:

folder = "/media/kevin/My Passport/2019-02-18-measurements/"
filename = "NaI-Ptw3"
movienumber = 2
Tframe = 1/200.  # In seconds the time per frame 

averageforspot = True  # Take average of 9 pixels for the spot of interest or not
hertz = 1
Tsignal = 1./hertz  # In seconds, used a bit
Asignal = 1.  # In volts, never used

doData.process(folder, filename, movienumber, Tframe, Tsignal, Asignal)


#%%
# Make a set of movies:
# copy paste as many as you want:

movieset = []  # First make an empty set

folder = "/media/kevin/My Passport/2019-02-18-measurements/"
filename = "NaI-Ptw3"
movienumber = 2
Tframe = 1/200.  # In seconds the time per frame 
Tsignal = 1./1  # In seconds, used a bit
Asignal = 1.  # In volts, never used actually
amovie = [folder, filename, movienumber, Tframe, Tsignal, Asignal]
movieset.append(amovie)


print(movieset)


