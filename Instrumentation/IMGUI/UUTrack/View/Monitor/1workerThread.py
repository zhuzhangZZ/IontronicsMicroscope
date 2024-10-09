"""
    UUTrack.View.Camera.workerThread
    ================================

    Thread that acquires continuously data until a variable is changed. This enables to acquire at any frame rate without
    freezing the GUI or overloading it with data being acquired too fast.
    .. sectionauthor:: Aquiles Carattino <aquiles@aquicarattino.com>
    .. sectionauthor:: Zhu Zhang <z.zhang@uu.nl>
"""

from pyqtgraph.Qt import QtCore

class workThread(QtCore.QThread):
    """Thread for acquiring from the camera. If the exposure time is long, this is
    needed to avoid freezing the GUI.
    """
    def __init__(self,_session,camera):
        QtCore.QThread.__init__(self)
        self._session = _session
        self.camera = camera
        self.origin = None
        self.keep_acquiring = True
    def __del__(self):
        self.wait()

    # def run(self):
    #     """ Triggers the Monitor to acquire a new Image.
    #     the QThread defined .start() method is a special method that sets up the thread and
    #     calls our implementation of the run() method.
    #     """
    #     first = True
    #     while self.keep_acquiring:
    #         if self.origin == 'snap':
    #             self.keep_acquiring = False
    #
    #         if self.keep_acquiring == False:
    #             self.camera.setAcquisitionMode(self.camera.MODE_SINGLE_SHOT)
    #         elif self.keep_acquiring == True:
    #             if first:
    #                 self.camera.setAcquisitionMode(self.camera.MODE_CONTINUOUS)
    #                 self.camera.triggerCamera() # Triggers the camera only once
    #                 first = False
    #         img = self.camera.readCamera()
    #
    #         self.emit(QtCore.SIGNAL('image'), img, self.origin)
    #     self.camera.stopAcq()
    #     return

    def run(self):
        """ Triggers the Monitor to acquire a new Image.
        the QThread defined .start() method is a special method that sets up the thread and
        calls our implementation of the run() method.
        """
        first = True
        while self.keep_acquiring:
            if self.origin == 'snap':
                self.keep_acquiring = False
            if first:
                self.camera.setAcquisitionMode(self.camera.MODE_CONTINUOUS)
                self.camera.triggerCamera()  # Triggers the camera only once
                first = True
            img = self.camera.readCamera()

            self.emit(QtCore.SIGNAL('image'), img, self.origin)
        self.camera.stopAcq()
        return




# from pyqtgraph.Qt import QtCore
# import threading
# import queue

# class WorkThread(QtCore.QThread):
    # """Thread for acquiring from the camera. If the exposure time is long, this is
    # needed to avoid freezing the GUI.
    # """
    # def __init__(self, _session, camera):
        # super().__init__()
        # self._session = _session
        # self.camera = camera
        # self.origin = None
        # self.keep_acquiring = True
        # self.image_queue = queue.Queue()

    # def __del__(self):
        # self.wait()

    # def run(self):
        # """ Triggers the Monitor to acquire a new Image. """
        # first = True

        # # Start a separate consumer thread to emit images
        # consumer_thread = threading.Thread(target=self.emit_images, daemon=True)
        # consumer_thread.start()

        # while self.keep_acquiring:
            # if self.origin == 'snap':
                # self.keep_acquiring = False
            # if first:
                # self.camera.setAcquisitionMode(self.camera.MODE_CONTINUOUS)
                # self.camera.triggerCamera()  # Triggers the camera only once
                # first = False
            # img = self.camera.readCamera()

            # # Add the image to the queue for processing by the consumer
            # self.image_queue.put(img)

        # self.camera.stopAcq()

    # def emit_images(self):
        # """ Consume images from the queue and emit them. """
        # while self.keep_acquiring or not self.image_queue.empty():
            # try:
                # # Attempt to get the next image from the queue
                # img = self.image_queue.get(timeout=1)
                # # Emit the signal using QtCore.SIGNAL
                # self.emit(QtCore.SIGNAL('image'), img, self.origin)
            # except queue.Empty:
                # pass
## https://chat.openai.com/share/9468fed6-5c67-4353-8818-9ff8bc2772dd
