import os
import numpy as np
# import matplotlib.pyplot as plt
# import softpotato as sp

import hardpotato.load_data as load_data
import hardpotato.save_data as save_data
import hardpotato.chi760e as chi760e
import hardpotato.chi1205b as chi1205b
import hardpotato.chi601e as chi601e
import hardpotato.chi1242b as chi1242b
import hardpotato.emstatpico as emstatpico

import hardpotato.pico_instrument as instrument
import hardpotato.pico_mscript as mscript
import hardpotato.pico_serial as serial

# Potentiostat models available: 
models_available = ['chi1205b', 'chi1242b', 'chi601e', 'chi760e', 'emstatpico']

# Global variables
folder_save = '.'
model_pstat = 'no pstat'
path_lib = '.'

class Test:
    '''
    '''
    def __init__(self):
        print('Test from potentiostat module')


class Info:
    '''
    '''

    def __init__(self, model):
        self.model = model
        if self.model == 'chi1205b':
            self.info = chi1205b.Info()
        elif self.model == 'chi1242b':
            self.info = chi1242b.Info()
        elif self.model == 'chi601e':
            self.info = chi601e.Info()
        elif self.model == 'chi760e':
            self.info = chi760e.Info()
        elif self.model == 'emstatpico':
            self.info = emstatpico.Info()
        else:
            print('Potentiostat model ' + model + ' not available in the library.')
            print('Available models:', models_available)

    def specifications(self):
        self.info.specifications()

class Setup:
    def __init__(self, model=0, path='.', folder='.', port=None, verbose=1):
        global folder_save
        folder_save = folder
        global model_pstat
        model_pstat = model
        global path_lib
        path_lib = path
        global port_
        port_ = port
        if verbose:
            self.info()

    def info(self):
        print('\n----------')
        print('Potentiostat model: ' + model_pstat)
        print('Potentiostat path: ' + path_lib)
        print('Save folder: ' + folder_save)
        print('----------\n')


class Technique:
    '''
    '''

    def __init__(self, text='', fileName='CV'):
        self.text = text # text to write as macro
        self.fileName = fileName
        self.technique = 'Technique'
        self.bpot = False

    def writeToFile(self):
        if model_pstat[0:3] == 'chi':
            file = open(folder_save + '/' + self.fileName + '.mcr', 'wb')
            file.write(self.text.encode('ascii'))
            file.close()
        elif model_pstat == 'emstatpico':
            file = open(folder_save + '/' + self.fileName + '.mscr', 'wb')
            file.write(self.text.encode('ascii'))
            file.close()

    def run(self):
        if model_pstat[0:3] == 'chi':
            self.message()
            # Write macro:
            self.writeToFile()
            # Run command:
            command = path_lib #+ '/chi760e.exe'
            param = ' /runmacro:\"' + folder_save + '/' + self.fileName + '.mcr\"'
            os.system(command + param)
            self.message(start=False)
            # self.plot()
            print("not plot figure")
        elif model_pstat == 'emstatpico':
            self.message()
            self.writeToFile()
            if port_ is None:
                self.port = serial.auto_detect_port()
            with serial.Serial(self.port,1) as comm:
                dev = instrument.Instrument(comm)
                dev.send_script(folder_save + '/' + self.fileName + '.mscr')
                result = dev.readlines_until_end()
            self.data = mscript.parse_result_lines(result)
            fileName = folder_save + '/' + self.fileName + '.txt'
            save = save_data.Save(self.data, fileName, self.header, model_pstat, 
                           self.technique, bpot=self.bpot)
            self.message(start=False)
            # self.plot()
            print("not plot figure")
        else:
            print('\nNo potentiostat selected. Aborting.')

    # def plot(self):
    #     figNum = np.random.randint(100) # To prevent rewriting the same plot
    #     if self.technique == 'CV':
    #         cv = load_data.CV(self.fileName+'.txt', folder_save, model_pstat)
    #         sp.plotting.plot(cv.E, cv.i, show=False, fig=figNum,
    #                          fileName=folder_save + '/' + self.fileName)
    #     elif self.technique == 'LSV':
    #         lsv = load_data.LSV(self.fileName+'.txt', folder_save, model_pstat)
    #         sp.plotting.plot(lsv.E, lsv.i, show=False, fig=figNum,
    #                          fileName=folder_save + '/' + self.fileName)
    #     elif self.technique == 'CA':
    #         ca = load_data.CA(self.fileName+'.txt', folder_save, model_pstat)
    #         sp.plotting.plot(ca.t, ca.i, show=False, fig=figNum,
    #                          xlab='$t$ / s', ylab='$i$ / A',
    #                          fileName=folder_save + '/' + self.fileName)
    #     elif self.technique == 'OCP':
    #         ocp = load_data.OCP(self.fileName+'.txt', folder_save, model_pstat)
    #         sp.plotting.plot(ocp.t, ocp.E, show=False, fig=figNum,
    #                          xlab='$t$ / s', ylab='$E$ / V',
    #                          fileName=folder_save + '/' + self.fileName)
    #     elif self.technique == 'SSF':
    #         ssf = load_data.SSF(self.fileName+'.txt', folder_save, model_pstat)
    #         sp.plotting.plot(ssf.E, ssf.i, show=False, fig=figNum,
    #                          fileName=folder_save + '/' + self.fileName)
    #     plt.close()
    #


    def message(self, start=True):
        if start:
            print('----------\nStarting ' + self.technique)
            if self.bpot:
                print('Running in bipotentiostat mode')
        else:
            print(self.technique + ' finished\n----------\n')

    def bipot(self, E=-0.2, sens=1e-6):
        if self.technique != 'OCP' and self.technique != 'EIS':
            if model_pstat == 'chi760e':
                self.tech.bipot(E, sens)
                self.text = self.tech.text
                self.bpot = True
            elif model_pstat == 'emstatpico':
                self.tech.bipot(E, sens)
                self.text = self.tech.text
                self.bpot = True
        else:
            print(self.technique + ' does not have bipotentiostat mode')
     

class CV(Technique):
    '''
    '''
    def __init__(self, Eini=-0.2, Ev1=0.2, Ev2=-0.2, Efin=-0.2, sr=0.1,
                 dE=0.001, nSweeps=2, sens=1e-6,
                 fileName='CV', header='CV', ExtTri = False, **kwargs):
        self.header = header
        if model_pstat == 'chi601e':
            self.tech = chi601e.CV(Eini, Ev1, Ev2, Efin, sr, dE, nSweeps, sens,
                               folder_save, fileName, header, path_lib,
                               **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'CV'
        if model_pstat == 'chi760e':
            self.tech = chi760e.CV(Eini, Ev1, Ev2, Efin, sr, dE, nSweeps, sens,
                               folder_save, fileName, header, path_lib, ExtTri,
                               **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'CV'
        elif model_pstat == 'chi1205b':
            self.tech = chi1205b.CV(Eini, Ev1, Ev2, Efin, sr, dE, nSweeps, sens,
                               folder_save, fileName, header, path_lib,
                               **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'CV'
        elif model_pstat == 'chi1242b':
            self.tech = chi1242b.CV(Eini, Ev1, Ev2, Efin, sr, dE, nSweeps, sens,
                               folder_save, fileName, header, path_lib,
                               **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'CV'
        elif model_pstat == 'emstatpico':
            self.tech = emstatpico.CV(Eini, Ev1, Ev2, Efin, sr, dE, nSweeps, sens, 
                                folder_save, fileName, header, path_lib='',
                                **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'CV'
        else:
            print('Potentiostat model ' + model_pstat + ' does not have CV.')



class LSV(Technique):
    '''
    '''
    def __init__(self, Eini=-0.2, Efin=0.2, sr=0.1, dE=0.001, sens=1e-6,
                 fileName='LSV', header='LSV', **kwargs):
        self.header = header
        if model_pstat == 'chi601e':
            self.tech = chi601e.LSV(Eini, Efin, sr, dE, sens, folder_save, fileName, 
                                header, path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'LSV' 
        if model_pstat == 'chi760e':
            self.tech = chi760e.LSV(Eini, Efin, sr, dE, sens, folder_save, fileName, 
                                header, path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'LSV'  
        elif model_pstat == 'chi1205b':
            self.tech = chi1205b.LSV(Eini, Efin, sr, dE, sens, folder_save, fileName, 
                                header, path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'LSV'  
        elif model_pstat == 'chi1242b':
            self.tech = chi1242b.LSV(Eini, Efin, sr, dE, sens, folder_save, fileName, 
                                header, path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'LSV'  
        elif model_pstat == 'emstatpico':
            self.tech = emstatpico.LSV(Eini, Efin, sr, dE, sens, folder_save, fileName, 
                                header, path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'LSV'
        else:
            print('Potentiostat model ' + model_pstat + ' does not have LSV.')



class CA(Technique):
    '''
    '''
    def __init__(self, Estep=0.2, dt=0.001, ttot=2, sens=1e-6,
                 fileName='CA', header='CA', **kwargs):
        self.header = header
        if model_pstat == 'chi601e':
            self.tech = chi601e.CA(Estep, dt, ttot, sens, folder_save, fileName,
                               header, path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'CA'
        if model_pstat == 'chi760e':
            self.tech = chi760e.CA(Estep, dt, ttot, sens, folder_save, fileName,
                               header, path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'CA'
        elif model_pstat == 'chi1205b':
            self.tech = chi1205b.CA(Estep, dt, ttot, sens, folder_save, fileName,
                               header, path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'CA'
        elif model_pstat == 'chi1242b':
            self.tech = chi1242b.CA(Estep, dt, ttot, sens, folder_save, fileName,
                               header, path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'CA'
        elif model_pstat == 'emstatpico':
            self.tech = emstatpico.CA(Estep, dt, ttot, sens, folder_save, fileName,
                               header, path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'CA'
        else:
            print('Potentiostat model ' + model_pstat + ' does not have CA.')




class OCP(Technique):
    '''
    '''
    def __init__(self, ttot=2, dt=0.01, fileName='OCP', header='OCP', **kwargs):
        self.header = header
        if model_pstat == 'chi601e':
            self.tech = chi601e.OCP(ttot, dt, folder_save, fileName, header, 
                                    path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'OCP'
        if model_pstat == 'chi760e':
            self.tech = chi760e.OCP(ttot, dt, folder_save, fileName, header, 
                                    path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'OCP'
        elif model_pstat == 'chi1205b':
            self.tech = chi1205b.OCP(ttot, dt, folder_save, fileName, header, 
                                     path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'OCP'
        elif model_pstat == 'chi1242b':
            self.tech = chi1242b.OCP(ttot, dt, folder_save, fileName, header, 
                                     path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'OCP'
        elif model_pstat == 'emstatpico':
            self.tech = emstatpico.OCP(ttot, dt, folder_save, fileName, header, 
                                       path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'OCP'
        else:
            print('Potentiostat model ' + model_pstat + ' does not have OCP.')

class NPV(Technique):
    '''
    '''
    def __init__(self, Eini=0.5, Efin=-0.5, dE=0.01, tsample=0.1, twidth=0.05, 
                 tperiod=10, sens=1e-6,
                 fileName='NPV', header='NPV performed with CHI760', **kwargs):
        if model_pstat == 'chi760e':
            self.tech = chi760e.NPV(Eini, Efin, dE, tsample, twidth, tperiod, sens,
                         folder_save, fileName, header, path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'NPV'
        elif model_pstat == 'chi601e':
            self.tech = chi601e.NPV(Eini, Efin, dE, tsample, twidth, tperiod, sens,
                         folder_save, fileName, header, path_lib, **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'NPV'
        else:
            print('Potentiostat model ' + model_pstat + ' does not have NPV.')



class EIS(Technique):
    '''
    '''
    def __init__(self, Eini=0, low_freq=1, high_freq=1000, amplitude=0.01, 
                 sens=1e-6, fileName='EIS', header='EIS', **kwargs):
        self.header = header
        if model_pstat == 'chi760e':
            self.tech = chi760e.EIS(Eini, low_freq, high_freq, amplitude, sens, qt, 
                                    folder_save, fileName, header, path_lib, 
                                    **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'EIS'
        else:
            print('Potentiostat model ' + model_pstat + ' does not have EIS.')


class SSF(Technique):
    '''
    '''
    def __init__(self, Eini= -1, dE =0.001, dt=0.01, sens=1e-6, Eswi1=-1, Eswf1=-1, sr1=0.1, Estep1=-1, tstep1=10, Eswi2=-1, Eswf2=0, sr2=0.1,
                 fileName='SSF', header='SSF', ExtTri = False, **kwargs):
        self.header = header
        if model_pstat == 'chi601e':
            self.tech = chi601e.SSF(Eini, dE, dt, sens, Eswi1, Eswf1, sr1, Estep1, tstep1, Eswi2, Eswf2, sr2,
                               folder_save, fileName, header, path_lib, ExtTri,
                               **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'SSF'
        if model_pstat == 'chi760e':
            self.tech = chi760e.SSF(Eini, dE, dt, sens, Eswi1, Eswf1, sr1, Estep1, tstep1, Eswi2, Eswf2, sr2,
                               folder_save, fileName, header, path_lib, ExtTri,
                               **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'SSF'
        elif model_pstat == 'chi1205b':
            self.tech = chi1205b.SSF(Eini, dE, dt, sens, Eswi1, Eswf1, sr1, Estep1, tstep1, Eswi2, Eswf2, sr2,
                               folder_save, fileName, header, path_lib, ExtTri,
                               **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'SSF'
        elif model_pstat == 'chi1242b':
            self.tech = chi1242b.SSF(Eini, dE, dt, sens, Eswi1, Eswf1, sr1, Estep1, tstep1, Eswi2, Eswf2, sr2,
                               folder_save, fileName, header, path_lib, ExtTri,
                               **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'SSF'
        elif model_pstat == 'emstatpico':
            self.tech = emstatpico.SSF(Eini, dE, dt, sens, Eswi1, Eswf1, sr1, Estep1, tstep1, Eswi2, Eswf2, sr2,
                               folder_save, fileName, header, path_lib = '',
                                **kwargs)
            Technique.__init__(self, text=self.tech.text, fileName=fileName)
            self.technique = 'SSF'
        else:
            print('Potentiostat model ' + model_pstat + ' does not have SSF.')


if __name__ == '__main__':
    sens = 1e-8
    sr = [0.1, 0.2, 0.5]
    folder = 'C:/Users/oliverrz/Desktop/Oliver/Data/220113_PythonMacros'
