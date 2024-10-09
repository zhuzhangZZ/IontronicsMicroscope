import hardpotato as hp
# import softpotato as sp
import os
import threading
import multiprocessing
from UUTrack.log import get_logger, log_to_file, log_to_screen
logger = get_logger(__name__)


class potentiostat():

    def __init__(self, model):
        super().__init__()
        self.ps_model = model  # potentiostat model 'chi760e'
        self.exePath = 'D:/Equipment/CHI/chi760e/chi760e.exe' # .exe software place

    def initializePotentialstat(self):
        """ Initializes the communication with the potentiostat.
        """
        self.qt =2
        self.cv_threading = None

        logger.debug('Initializing CHI PotentialStat Camera:', self.ps_model)

    def setCV(self, Eini = 0, Ev1 = 0, Ev2 = -1, Efin = 0, sr = 0.05, dE = 0.001, nSweeps = 2, sens = 1e-4 ,
              fileName = 'video1_m0', saveFolder = "C:/" ,  header = 'CV', ExtTri = True):

        self.folder = saveFolder  # folder to save the CV data
        # self.folder = 'C:/Data/Zhu/2023-08-11_Pd'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            print('the folder of %s has created' % self.folder)
        else:
            print('the folder of %s already exists ' % self.folder)
        # Initialization:
        hp.potentiostat.Setup(self.ps_model, self.exePath, self.folder)
        # Experimental parameters:
        self.ExtTri = ExtTri
        self.Eini = Eini  # V, initial potential
        self.Ev1 = Ev1  # V, first vertex potential
        self.Ev2 = Ev2  # V, second vertex potential
        self.Efin = Efin  # V, finad potential
        self.sr = sr  # V/s, scan rate
        self.dE = dE  # V, potential increment
        self.nSweeps = nSweeps  # number of sweeps
        self.sens = sens  # A/V, current sensitivity
        # E2 = 0.5        # V, potential of the second working electrode
        # sens2 = 1e-9    # A/V, current sensitivity of the second working electrode
        self.fileName = fileName  # base file name for data file
        self.header = header  # header zfor data file
        self.cv = hp.potentiostat.CV(self.Eini, self.Ev1, self.Ev2, self.Efin, self.sr, self.dE, self.nSweeps, self.sens,
                                self.fileName, self.header, self.ExtTri)
    def runCV(self):
        self.cv.run()
        # Run experiment:
        # def runCV_thread():
        #
        #     self.cv.run()
        #     # Load recently acquired data
        #     # data = hp.load_data.CV(self.fileName + '.txt', self.folder, self.ps_model)
        #     # self.i = data.i
        #     # self.E = data.E
        #     # Plot CV with softpotato
        #
        # # self.cv_threading = threading.Thread(target = runCV_thread)
        # self.cv_threading = multiprocessing.Process(target=runCV_thread)
        # self.cv_threading.start()

        # if not self.cv_threading.is_alive():
        #     sp.plotting.plot(E, i, fig=1, show=1)


    def setSSF(self, Eholding = -1, holdingTime = 10, Eini = 0, dE=0.001, dt=0.1, sens=0.004, Eend=0, sr =0.05,
                    fileName = "video_m0", saveFolder = "C:/" , header = 'SSF', ExtTri = True, qt = 2):

        self.folder = saveFolder  # folder to save the CV data
        # self.folder = 'C:/Data/Zhu/2023-08-11_Pd'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            print('the folder of %s has created' % self.folder)
        else:
            print('the folder of %s already exists ' % self.folder)
        # Initialization:
        hp.potentiostat.Setup(self.ps_model, self.exePath, self.folder)
        # Experimental parameters:

        self.ExtTri = ExtTri
        # tech=ssf  #select Sweep-Step Functions
        self.Eini = Eini  # initial potential in V
        self.dE = dE  # sweep sample interval in V
        self.dt = dt  # step sample interval in s
        self.qt = qt  # quiescent time before run in s
        self.sens = sens  # sensitivity in A/V
        self.Eswi1 = Eholding  # initial potential in V for Sequence 1: Sweep
        self.Eswf1 = Eholding  # final potential in V for Sequence 1: Sweep
        self.sr1 = sr  # 1e-4   -   10 scan rate in V/s for Sequence 1: Sweep
        self.Estep1 = Eholding  # -10   -   +10 step potential in V for Sequence 2: Step
        self.tstep1 = holdingTime  # 0   -   10000 step time in s for Sequence 2: Step
        self.Eswi2 = Eholding  # -10  -  +10 initial potential in V for Sequence 3: Sweep
        self.Eswf2 = Eend  # 10   -   +10 final potential in V for Sequence 3: Sweep
        self.sr2 = sr  # -   10 scan rate in V/s for Sequence 3: Sweep
        self.fileName = fileName  # base file name for data file
        self.header = header  # header for data file

        self.ssf = hp.potentiostat.SSF(self.Eini, self.dE, self.dt, self.sens, self.Eswi1, self.Eswf1, self.sr1, self.Estep1, \
                                  self.tstep1, self.Eswi2, self.Eswf2, self.sr2, self.fileName, self.header,
                                  self.ExtTri, qt=self.qt)

    def runSSF(self):

        # Run experiment:
        self.ssf.run()
        # Load recently acquired data
        # data = hp.load_data.SSF(self.fileName + '.txt', self.folder, self.ps_model)
        # i = -data.i
        # E = data.E
        # Plot CV with softpotato
        # sp.plotting.plot(E, i, fig=1, show=1)