
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PySide6.QtCore import QFile, QIODevice, QTimer, Signal, Qt, QThread 
from PySide6.QtGui import QIntValidator
from PySide6.QtUiTools import QUiLoader

from narsil.liverun.ui.ui_ViewerWindow import Ui_ViewerWindow

from pathlib import Path
from skimage import io

import numpy as np
import sys


class ViewerWindow(QMainWindow):


    def __init__(self, saveDir=None, database=None):
        super(ViewerWindow, self).__init__()
        self.ui = Ui_ViewerWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Dead-Alive Viewer Window")

        #
        self.saveDir = saveDir
        
        self.database = database

        # thread object that will fetch 
        self.imageFetchThread = None

        self.currentPosition = None
        self.positionNoValidator = QIntValidator(0, 10000, self.ui.positionNoLine)
        self.channelNoValidator = QIntValidator(0, 200, self.ui.channelNoLine)
        self.currentChannelNo = None

        self.exptRunning = False

        self.showPhase = True
        self.showSeg = False


        self.ui.activePositions.setSortingEnabled(True)

        self.activePositions = []

        self.sentPositions = []


        self.setupButtonHandlers()

    def setupButtonHandlers(self,):
        # adjust plotting stuff
        self.ui.imagePlot.ui.histogram.hide()
        self.ui.imagePlot.ui.roiBtn.hide()
        self.ui.imagePlot.ui.menuBtn.hide()


        # set positionNo validator
        self.ui.positionNoLine.setValidator(self.positionNoValidator)
        self.ui.positionNoLine.textChanged.connect(self.positionChanged)

        # set channelNo validator
        self.ui.channelNoLine.setValidator(self.channelNoValidator)
        self.ui.channelNoLine.textChanged.connect(self.channelNoChanged)

        # set option handlers to show phase or seg
        self.ui.phaseImage.toggled.connect(self.setImageType)
        self.ui.cellSegImage.toggled.connect(self.setImageType)

        # fetch button handler 
        self.ui.fetchButton.clicked.connect(self.fetchData)

        # set expt running options useful for not doing database connections over and over again
        self.ui.isExptRunning.toggled.connect(self.setExptRunning)

        # rolling window width

        # area threshold slider handler

        # length threshold slider handler

        # no of cell like objects slider

        # Update filter parameters button

        # find all tweezalbe channels button

        # show position button?

        # remove position button

        # undo position button

        # reset position button

        # next auto

        # send tweeze positions to main window button 
    
    def positionChanged(self):
        position = self.ui.positionNoLine.text()
        try:
            intPosition = int(position)
        except:
            self.ui.positionNoLine.setText("")
            intPosition = None
        finally:
            self.currentPosition = intPosition
        sys.stdout.write(f"Position set to {self.currentPosition}\n")
        sys.stdout.flush()

    def channelNoChanged(self):
        channelNo = self.ui.channelNoLine.text()
        try:
            intChannelNo  = int(channelNo)
        except:
            self.ui.channelNoLine.setText("")
            intChannelNo = None
        finally:
            self.currentChannelNo = intChannelNo
        sys.stdout.write(f"Channel no set to {self.currentChannelNo}\n")
        sys.stdout.flush()
    
    def setImageType(self, clicked):
        self.showSeg = self.ui.cellSegImage.isChecked()
        self.showPhase = self.ui.phaseImage.isChecked()
        sys.stdout.write(f"Phase: {self.showPhase} Seg: {self.showSeg}\n")
        sys.stdout.flush()

    def setExptRunning(self, buttonState):
        self.exptRunning = buttonState
        sys.stdout.write(f"Expt is running : {self.exptRunning}\n")
        sys.stdout.flush()

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
        self.ui.imagePlot.setImage(self.imageFetchThread.data, autoLevels=True)
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

