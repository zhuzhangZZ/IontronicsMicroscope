"""
    UUTrack.View.Monitor.configWidget.py
    ===================================
    Simple widget for storing the parameters of the :mod:`UUTrack.Model._session`.
    It creates and populates tree thanks to the :meth:`UUTrack.Model._session._session.getParams`.
    The widget has two buttons, one that updates the session by emitting a `signal` to the main thread and another the repopulates the tree whith the available parameters.

    .. todo:: Remove the printing to screen of the parameters once the debugging is done.

    .. sectionauthor:: Aquiles Carattino <aquiles@aquicarattino.com>
"""

from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem
from pyqtgraph.widgets import ComboBox

class configWidget(QtGui.QWidget):
    """Widget for configuring the main parameters of the camera.
    """
    def __init__(self, session, parent=None):
        QtGui.QWidget.__init__(self, parent)

        # Set the background color
        # self.setAutoFillBackground(True)
        # p = self.palette()
        # p.setColor(self.backgroundRole(), QtGui.QColor('black'))
        # self.setPalette(p)

        self._session = session.copy()
        self._session_new = self._session.copy()  # To store the changes until applied
        self.t = ParameterTree()
        # self.t.setStyleSheet("QTreeView { background-color: grey; color: red; }")  # Set stylesheet for background color

        self.populateTree(session)
        self.layout = QtGui.QGridLayout()

        # self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.t, 0, 0, 1, 2)
        # self.t.setStyleSheet("background-color: grey; color: red;")
        self.apply = QtGui.QPushButton('Apply')
        self.cancel = QtGui.QPushButton('Cancel')

        # self.apply.setStyleSheet("background-color: grey; color: white;")  # Set background color for Apply button
        # self.cancel.setStyleSheet("background-color: grey; color: white;")  # Set background color for Cancel button

        self.apply.clicked.connect(self.updateSession)
        self.cancel.clicked.connect(self.populateTree)

        self.layout.addWidget(self.apply, 1, 0)
        self.layout.addWidget(self.cancel, 1, 1)
        self.setLayout(self.layout)


        #
    def change(self, param, changes):
        """Updates the values while being updated"""
        for param, change, data in changes:
            to_update = param.name().replace(' ','_')
            path = self.p.childPath(param)[0]
            self._session_new.params[path][to_update] = data
            if self._session.Debug['to_screen']:
                print(self._session_new.params['Camera']['roi_x1'])
                print(self._session_new.params['Camera']['roi_x2'])
                print(self._session_new.params['Camera']['roi_y1'])
                print(self._session_new.params['Camera']['roi_y2'])
                print(self._session.params['Camera']['roi_x1'])
                print(self._session.params['Camera']['roi_x2'])
                print(self._session.params['Camera']['roi_y1'])
                print(self._session.params['Camera']['roi_y2'])

    def updateSession(self):
        """ Updates the session and sends a signal"""
        self._session = self._session_new.copy()
        self.emit(QtCore.SIGNAL('updateSession'), self._session)

    def populateTree(self, session=0):
        """Fills the tree with the values from the Session"""
        # self.t.setStyleSheet("background-color: #AAAAAA; color: red;")
        if type(session) != type(0):
            self._session = session.copy()
            self._session_new = session.copy()
        params = self._session.getParams()
        self.p = Parameter.create(name='params', type='group', children=params)
        self.p.sigTreeStateChanged.connect(self.change)

        self.t.setParameters(self.p, showTop=False)
