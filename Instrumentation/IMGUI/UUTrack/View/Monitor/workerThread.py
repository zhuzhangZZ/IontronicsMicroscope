from pyqtgraph.Qt import QtCore
import zmq
import threading

class workThread(QtCore.QThread):
    """Thread for acquiring from the camera. If the exposure time is long, this is
    needed to avoid freezing the GUI.
    """
    def __init__(self, _session, camera, zmq_context=None):
        super().__init__()
        self._session = _session
        self.camera = camera
        self.origin = None
        self.keep_acquiring = True

        # Set up ZeroMQ context and sockets
        self.zmq_context = zmq_context or zmq.Context()
        self.push_socket = self.zmq_context.socket(zmq.PUSH)
        self.pull_socket = self.zmq_context.socket(zmq.PULL)

        # Bind the PUSH socket to an in-process address
        self.push_socket.bind("inproc://camera_push")
        self.pull_socket.connect("inproc://camera_push")

    def __del__(self):
        self.wait()

    def run(self):
        """ Triggers the Monitor to acquire a new Image. """
        first = True

        # Start a separate consumer thread to emit images
        consumer_thread = threading.Thread(target=self.emit_images, daemon=True)
        consumer_thread.start()

        while self.keep_acquiring:
            if self.origin == 'snap':
                self.keep_acquiring = False
            if first:
                self.camera.setAcquisitionMode(self.camera.MODE_CONTINUOUS)
                self.camera.triggerCamera()  # Triggers the camera only once
                first = False
            img = self.camera.readCamera()

            # Send the image through the PUSH socket
            self.push_socket.send_pyobj((img, self.origin))

        self.camera.stopAcq()

    def emit_images(self):
        """ Consume images from the PULL socket and emit them. """
        while self.keep_acquiring:
            try:
                # Receive the image from the PULL socket
                img, origin = self.pull_socket.recv_pyobj(flags=zmq.NOBLOCK)
                # Emit the signal using QtCore.SIGNAL
                self.emit(QtCore.SIGNAL('image'), img, origin)
            except zmq.Again:
                pass
