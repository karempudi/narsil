
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
        self.saveDir = Path("/home/pk/Documents/realtimeData/analysisData/")
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


        self.ui.activePositionsList.setSortingEnabled(True)
        self.activePositions = []


        self.rollingWindow = None
        self.rollingWindowValidator = QIntValidator(0, 99, self.ui.windowLengthLine)
        self.areaThreshold = None
        self.lengthThreshold = None
        self.noCellObjects = None

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
        self.ui.windowLengthLine.setValidator(self.rollingWindowValidator)
        self.ui.windowLengthLine.textChanged.connect(self.setRollingWindow)

        # area threshold slider handler

        # length threshold slider handler

        # no of cell like objects slider

        # Update filter parameters button

        # find all tweezalbe channels button
        self.ui.findLocationsButton.clicked.connect(self.getAllLocationsFake)


        # hook up if the selection changed
        self.ui.activePositionsList.itemSelectionChanged.connect(self.showCurrentPosition)

        # show position button?
        self.ui.showButton.clicked.connect(self.showCurrentPosition)

        # remove position button
        self.ui.removeButton.clicked.connect(self.removeCurrentPosition)

        # undo position button
        self.ui.undoButton.clicked.connect(self.undoRemovedPosition)

        # reset position button
        self.ui.resetButton.clicked.connect(self.resetAllPositions)

        # next auto
        self.ui.nextAutoButton.clicked.connect(self.nextAutoPosition)

        # send tweeze positions to main window button 
        self.ui.sendTweezePositionsButton.clicked.connect(self.sendTweezableToMain)
    
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
        imgType = 'phase' if self.showPhase else 'cellSeg'

        if self.imageFetchThread is None:
            self.imageFetchThread =  ImageFetchThread({'positionNo': self.currentPosition, 
                                'channelNo': self.currentChannelNo,
                                'type': imgType,
                                'dir': self.saveDir})
            self.imageFetchThread.start()
            self.imageFetchThread.dataFetched.connect(self.updateImage)

    def setRollingWindow(self):
        rollingWindow = self.ui.windowLengthLine.text()
        try:
            intWindow = int(rollingWindow)
        except:
            self.ui.windowLengthLine.setText("")
            intWindow = None
        finally:
            self.rollingWindow = intWindow
        sys.stdout.write(f"Rolling window length : {self.rollingWindow}\n")
        sys.stdout.flush()


    def getAllLocationsFake(self, clicked):
        # populate the list widget will all possible locations
        position = 3
        for channelNo in range(0, 100):
            item = "Pos: "  + str(position) + " Ch: " + str(channelNo)
            self.ui.activePositionsList.addItem(item)
            self.activePositions.append((position, channelNo))

        if len(self.activePositions) > 0:
            self.ui.activePositionsList.setCurrentRow(0)

    def showCurrentPosition(self, clicked=None):
        try:
            selectedItem = self.ui.activePositionsList.currentItem().text()
            position = int(selectedItem.split(" ")[1])
            channelNo = int(selectedItem.split(" ")[3])
            sys.stdout.write(f"Selected row is -- {position} -- {channelNo} \n")
            sys.stdout.flush()
        except:
            self.currentPosition = None
            self.currentChannelNo = None
            sys.stdout.write("Selection couldn't be got \n")
            sys.stdout.flush()
        finally:
            self.currentPosition = position
            self.currentChannelNo = channelNo
            self.fetchData()

    def removeCurrentPosition(self, clicked):
        pass

    def undoRemovedPosition(self, clicked):
        pass

    def resetAllPositions(self, clicked):
        pass

    def nextAutoPosition(self, clicked):
        pass 

    def sendTweezableToMain(self, clicked):
        pass

    def getAllLocations(self):
        # populate the list widget with possible tweeze locations
        pass

    def updateImage(self):
        sys.stdout.write("Image received\n")
        sys.stdout.flush()
        self.ui.imagePlot.setImage(self.imageFetchThread.data, autoLevels=True)
        sys.stdout.write("Image plotted\n")
        sys.stdout.flush()
        self.imageFetchThread = None

class ImageFetchThread(QThread):

    dataFetched = Signal()

    def __init__(self, fetch_data):
        super(ImageFetchThread, self).__init__()
        self.fetch_data = fetch_data
        self.data = None

    def run(self):
        # run code and fetch stuff
        sys.stdout.write(f"Fetch image thread to get position: {self.fetch_data['positionNo']} and channelNo: {self.fetch_data['channelNo']} \n")
        sys.stdout.flush()
        try:
            # fetch all the images that are there in the directory
            # construct image by fetching
            if self.fetch_data['type'] == 'phase':
                directory = self.fetch_data['dir'] / str(self.fetch_data['positionNo']) / "oneMMChannelPhase" / str(self.fetch_data['channelNo'])
            elif self.fetch_data['type'] == 'cellSeg':
                directory = self.fetch_data['dir'] / str(self.fetch_data['positionNo']) / "oneMMChannelCellSeg" / str(self.fetch_data['channelNo'])

            number_images = 20
            
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
