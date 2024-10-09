"""
    UUTrack.View.CHIIcontrol
    =========================
    Holds all communication necessary with the CHI potentiostat.
    Uses the python package "hardpatato"
    for the functions details, refer to https://github.com/jrlLAB/hardpotato
    .. sectionauthor:: Zhu Zhang <z.zhang@uu.nl>
"""

from UUTrack.Model.Potentiostats import CHI760e
from time import sleep
import threading
import subprocess
import os

class CHI760ePS():
    def __init__(self, ps_model = 'chi760e', parent=None):
        # Reset device
        self.PScontrol = CHI760e.potentiostat(ps_model)
        # Remember parent
        self.parent = parent
        # Create tasks defined above:
        # Create variables used for keeping track of saving:
        self.sn = None  # "SaveName"
        self.movieN = 0
        # CHI control parameters
        self.psRUN = False
        self.auto_kick_cam = True
        self.ExtTri = True
        self.process = None
    def togglePStask(self, session):
        self._session = session
        if self.psRUN:
            # Turn switch and return:
            self.psRUN = False
            # Autostop measurement:
            if self.auto_kick_cam != 0:
                self.parent.movieSaveStop()
            return
        # Update settings:
        self.saveFolder = self._session.Saving['directory']
        if self.sn != self._session.Saving['filename_video']:
            self.sn = self._session.Saving['filename_video']
            self.movieN = 0
        else:
            self.movieN += 1
        fileName = (self.sn + "_m" + str(self.movieN))

        self.Tech = self._session.Potentiostat['Tech']
        self.ExtTri = self._session.Potentiostat['ExtTri']
        self.auto_kick_cam = int(self._session.Potentiostat['kick_cam'])
        if self.Tech == 'CV':
            self.Eini = self._session.CV['Eini']
            self.Ev1 = self._session.CV['Ev1']
            self.Ev2 = self._session.CV['Ev2']
            self.Efin = self._session.CV['Efin']
            self.sr = self._session.CV['SR']
            self.dE = self._session.CV['dE']
            self.nSweeps = int(abs(self._session.CV['nSweeps']))
            self.sens = self._session.CV['sens']

            self.PScontrol.setCV(Eini=self.Eini, Ev1=self.Ev1, Ev2=self.Ev2, Efin=self.Efin, sr=self.sr, dE=self.dE,
                                 nSweeps=self.nSweeps, sens=self.sens, fileName=fileName, saveFolder=self.saveFolder,
                                 header='CV',ExtTri=self.ExtTri)
        elif self.Tech == "SSF":
            self.Eini = self._session.SSF['Eini']
            self.Eholding = self._session.SSF['Eholding']
            self.holdingTime = self._session.SSF['holdingTime']
            self.Eend = self._session.SSF['Eend']
            self.sr = self._session.SSF['SR']
            self.dE = self._session.SSF['dE']
            self.sens = self._session.SSF['sens']
            self.dt = self._session.SSF['dt']
            self.qt = self._session.SSF['qt']
            self.PScontrol.setSSF(Eholding= self.Eholding, holdingTime=self.holdingTime, Eini= self.Eini, dE=self.dE,
                                  dt=self.dt, sens=self.sens, Eend=self.Eend, sr=self.sr, fileName=fileName, saveFolder=self.saveFolder,
                                  header='SSF', ExtTri=self.ExtTri, qt=self.qt)

        else:
            print('Technique for potential is not valid, change to default CV')
            self.Eini = self._session.CV['Eini']
            self.Ev1 = self._session.CV['Ev1']
            self.Ev2 = self._session.CV['Ev2']
            self.Efin = self._session.CV['Efin']
            self.sr = self._session.CV['SR']
            self.dE = self._session.CV['dE']
            self.nSweeps = int(abs(self._session.CV['nSweeps']))
            self.sens = self._session.CV['sens']
            self.PScontrol.setCV(Eini=self.Eini, Ev1=self.Ev1, Ev2=self.Ev2, Efin=self.Efin, sr=self.sr, dE=self.dE,
                                 nSweeps=self.nSweeps, sens=self.sens, fileName=fileName, saveFolder=self.saveFolder,
                                 header='CV', ExtTri=self.ExtTri)
        def runPS_thread():
            if self.Tech == 'CV':
                self.PScontrol.runCV()
            elif self.Tech == 'SSF':
                self.PScontrol.runSSF()
            else:
                print('Technique for potential is not valid, change to default CV')
                self.PScontrol.runCV()

        self.cv_threading = threading.Thread(target = runPS_thread)
        # self.cv_threading = multiprocessing.Process(target=runCV_thread)
        self.cv_threading.start()
        sleep(2 + 3)  # wait for launch the potentiostat and wait for the qt time
        self.psRUN = True
        # Autostart movieSave:
        if self.auto_kick_cam != 0:
            self.parent.movieSave()
        
    def CHI_movieSaver_control(self):
        if not self.psRUN:
            return

        # elif not self.PScontrol.cv_threading.is_alive():
        elif not self.cv_threading.is_alive():
            self.togglePStask(None)
            self.cv_threading.join()
            command = self.saveFolder + '/' + self.sn +"_m"+str(self.movieN) + '.bin'

            def openCV_thread():
                os.system(command)
            # """"
            # def openCV_thread():
            #     # Construct the command
            #     command = f"{self.saveFolder}/{self.sn}_m{str(self.movieN)}.bin"
            #
            #     # Terminate the existing process if it's still running
            #     if self.process and self.process.poll() is None:
            #         # self.process.terminate()
            #         self.process.kill()
            #     # Start a new process
            #     self.process = subprocess.Popen(command, shell=True)
            # # """s
            self.showCV_threading = threading.Thread(target=openCV_thread)
            # self.cv_threading = multiprocessing.Process(target=runCV_thread)
            self.showCV_threading.start()
            print('finished!CV!!')
            # self.PScontrol.cv_threading.join()


