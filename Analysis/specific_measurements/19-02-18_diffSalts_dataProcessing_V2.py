# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 09:50:16 2019

@author: kevin
"""

# Fow now: set working directory to "PDSM/Analysis/data_processing" to load, or put these files in the same folder
import dataProcessing_automated_V2 as doData


#%%
# Make a set of movies:
# copy paste as many as you want:

movieset = []  # First make an empty set

folder = "/media/kevin/My Passport/2019-02-18-measurements/"
Tframe = 1/200.  # In seconds the time per frame 
Tsignal = 1./1  # In seconds, used a bit
Asignal = 2.  # In volts, never used actually

savefolderbase = "/home/kevin/Documents/PDSM_data/2019-02-18_1Hz_2Vpp_V2-larger-maxFFT/"

filename = "DI-Pt"
movienumber = 1
savefolder = savefolderbase+filename
amovie = [folder, filename, movienumber, Tframe, Tsignal, Asignal, savefolder]
movieset.append(amovie)

filename = "NaCl-Pt"
movienumber = 2
savefolder = savefolderbase+filename
amovie = [folder, filename, movienumber, Tframe, Tsignal, Asignal, savefolder]
movieset.append(amovie)

filename = "DI-Ptw2"
movienumber = 3
savefolder = savefolderbase+filename
amovie = [folder, filename, movienumber, Tframe, Tsignal, Asignal, savefolder]
movieset.append(amovie)

filename = "NaBr-Ptw2"
movienumber = 2
savefolder = savefolderbase+filename
amovie = [folder, filename, movienumber, Tframe, Tsignal, Asignal, savefolder]
movieset.append(amovie)

filename = "NaDI-Ptw3"
movienumber = 2
savefolder = savefolderbase+filename
amovie = [folder, filename, movienumber, Tframe, Tsignal, Asignal, savefolder]
movieset.append(amovie)

filename = "NaI-Ptw3"
movienumber = 2
savefolder = savefolderbase+filename
amovie = [folder, filename, movienumber, Tframe, Tsignal, Asignal, savefolder]
movieset.append(amovie)


#%%
# Do all the processing in a row:

for i in movieset:
    print(i)
    doData.process(i[0],i[1],i[2],i[3],i[4],i[5],i[6])
    
    
    
