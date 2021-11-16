
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PySide6.QtCore import QFile, QIODevice, QTimer, Signal, Qt, QThread
from PySide6.QtUiTools import QUiLoader

from narsil.liverun.ui.ui_ViewerWindow import Ui_ViewerWindow

from pathlib import Path
from skimage import io

import numpy as np
import sys


class ViewerWindow(QMainWindow):


    def __init__(self):
        super(ViewerWindow, self).__init__()
        self.ui = Ui_ViewerWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Dead-Alive Viewer Window")

        # thread object that will fetch 
        self.imageFetchThread = None

        self.setupButtonHandlers()

    def setupButtonHandlers(self,):
        # fetch button handler 
        self.ui.fetchButton.clicked.connect(self.fetchData)


    def fetchData(self):
        # create a thread and call to get the data, once the data fetching
        # is done call the plotter
        sys.stdout.write("Fetch Button clicked\n")
        sys.stdout.flush()
        self.imageFetchThread =  ImageFetchThread({'position': 100, 'time': 1})
        self.imageFetchThread.start()
        self.imageFetchThread.dataFetched.connect(self.updateImage)

    def updateImage(self):
        sys.stdout.write("Image received\n")
        sys.stdout.flush()
        self.ui.graphicsView.ui.histogram.hide()
        self.ui.graphicsView.ui.roiBtn.hide()
        self.ui.graphicsView.ui.menuBtn.hide()
        self.ui.graphicsView.setImage(self.imageFetchThread.data, autoLevels=True)
        sys.stdout.write("Image plotted\n")
        sys.stdout.flush()

class ImageFetchThread(QThread):

    dataFetched = Signal()

    def __init__(self, fetch_data):
        super(ImageFetchThread, self).__init__()
        self.fetch_data = fetch_data
        self.data = None

    def run(self):
        # run code and fetch stuff
        sys.stdout.write(f"Fetch image thread to get position: {self.fetch_data['position']} and time: {self.fetch_data['time']} \n")
        sys.stdout.flush()
        try:
            number_images = np.random.randint(low=0, high=39)
            # construct image by fetching
            directory = Path("/home/pk/Documents/realtimeData/analysisData/3/oneMMChannelPhase/10/")
            
            files = [ directory / (str(i) + '.tiff') for i in range(0, number_images)]

            image = io.imread(files[0])
            for i in range(1, number_images):
                image = np.concatenate((image, io.imread(files[i])), axis = 1)
            
            self.data = image

            sys.stdout.write(f"Image shape: {image.shape}\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"Data couldnot be fetched : {e}\n")
            sys.stdout.flush()
            self.data = np.random.normal(loc=0.0, scale=1.0, size=(100, 100))

        #self.data = np.random.normal(loc=0.0, scale=1.0, size=(100, 100))
        self.dataFetched.emit()

    def getData(self):
        return self.data

