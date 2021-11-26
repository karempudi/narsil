
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PySide6.QtCore import QFile, QIODevice, QTimer, Signal, Qt, QThread 
from PySide6.QtGui import QIntValidator
from PySide6.QtUiTools import QUiLoader

from narsil.liverun.ui.ui_ViewerWindow import Ui_ViewerWindow

from pathlib import Path
from skimage import io

import numpy as np
import sys
import glob


class ViewerWindow(QMainWindow):


    def __init__(self, saveDir='/home/pk/Documents/realtimeData/analysisData/',
                database={
                    'dbname': 'exp21bp000',
                    'dbuser': 'postgres',
                    'dbpassword': 'postgres',
                    'tables': ['arrival', 'segment', 'deadalive', 'growth']
                }):
        super(ViewerWindow, self).__init__()
        self.ui = Ui_ViewerWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Dead-Alive Viewer Window")

        #
        self.saveDir = Path(saveDir)
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

        self.show20Images = True
        self.showAllImages = False


        self.ui.activePositionsList.setSortingEnabled(True)
        self.ui.tweezePositionsList.setSortingEnabled(True)
        self.showActivePositions = True
        self.activePositions = []
        self.tweezePositions = []


        self.rollingWindow = None
        self.rollingWindowValidator = QIntValidator(0, 99, self.ui.windowLengthLine)
        self.areaThreshold = None
        self.lengthThreshold = None
        self.noCellObjects = None

        self.sentPositions = []


        self.setupButtonHandlers()
    
    def setSaveDir(self, saveDirPath):
        self.saveDir = Path(saveDirPath)

    def setDatabase(self, database):
        self.database = database

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

        # set if you want 20 images or all images
        self.ui.getOnly20Radio.toggled.connect(self.setNumberOfImagesToGet)
        self.ui.getAllImagesRadio.toggled.connect(self.setNumberOfImagesToGet)

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

        # hook up if selection changed
        self.ui.tweezePositionsList.itemSelectionChanged.connect(self.showCurrentPosition)

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

        # select which position list to display
        self.ui.viewActiveListCheck.toggled.connect(self.setViewActivePositions)

        # move position and channel item to possible tweeze positions list
        self.ui.toTweezeListButton.clicked.connect(self.sendPositionToTweezeList)

        # move position and channel item back to active positions list
        self.ui.toActiveListButton.clicked.connect(self.sendPositionToActiveList)

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
        self.showCurrentPosition()

    def setNumberOfImagesToGet(self, clicked):
        self.show20Images = self.ui.getOnly20Radio.isChecked()
        self.showAllImages = self.ui.getAllImagesRadio.isChecked()
        sys.stdout.write(f"Getting 20 images: {self.show20Images} all Images: {self.showAllImages}\n")
        sys.stdout.flush()

        self.showCurrentPosition()

    def setExptRunning(self, buttonState):
        self.exptRunning = buttonState
        sys.stdout.write(f"Expt is running : {self.exptRunning}\n")
        sys.stdout.flush()

    def setViewActivePositions(self, buttonState):
        self.showActivePositions = buttonState
        self.showCurrentPosition()
        sys.stdout.write(f"Showing active positions: {self.showActivePositions}\n")
        sys.stdout.flush()

    def fetchData(self):
        # create a thread and call to get the data, once the data fetching
        # is done call the plotter
        sys.stdout.write("Fetch Button clicked\n")
        sys.stdout.flush()
        imgType = 'phase' if self.showPhase else 'cellSeg'
        numberOfImageToShow = True if self.show20Images else False

        if self.imageFetchThread is None:
            self.imageFetchThread =  ImageFetchThread({'positionNo': self.currentPosition, 
                                'channelNo': self.currentChannelNo,
                                'type': imgType,
                                'dir': self.saveDir,
                                'show20Images': numberOfImageToShow})
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
            if self.showActivePositions:
                selectedItem = self.ui.activePositionsList.currentItem().text()
            else:
                selectedItem = self.ui.tweezePositionsList.currentItem().text()

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
            self.ui.positionNoLine.setText(str(self.currentPosition))
            self.ui.channelNoLine.setText(str(self.currentChannelNo))
            if (self.currentPosition  is not None) and (self.currentChannelNo is not None):
                self.fetchData()

    def removeCurrentPosition(self, clicked):
        try:
            selectedItems = self.ui.activePositionsList.selectedItems()
            for item in selectedItems:
                self.ui.activePositionsList.takeItem(self.ui.activePositionsList.row(item))
                itemText = item.text()
                position = int(itemText.split(" ")[1])
                channelNo = int(itemText.split(" ")[3])
                self.activePositions.remove((position, channelNo))
                sys.stdout.write(f"Deleting {itemText}\n")
                sys.stdout.flush()
        except:
            sys.stdout.write(f"Nothing to remove\n")
            sys.stdout.flush()
        finally:
            if len(self.activePositions) == 0:
                self.currentPosition = None
                self.currentChannelNo = None

    def sendPositionToTweezeList(self, clicked):
        try:
            selectedItems = self.ui.activePositionsList.selectedItems()
            for item in selectedItems:
                self.ui.activePositionsList.takeItem(self.ui.activePositionsList.row(item))
                itemText = item.text()
                position = int(itemText.split(" ")[1])
                channelNo = int(itemText.split(" ")[3])
                self.ui.tweezePositionsList.addItem(itemText)
                self.activePositions.remove((position, channelNo))
                self.tweezePositions.append((position, channelNo))
                sys.stdout.write(f"Moved Pos: {position} ChNo: {channelNo} to tweeze list\n")
                sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"Moving channel to tweeze list failed -- {e}\n")
            sys.stdout.flush()

        finally:
            if len(self.tweezePositions) > 0:
                self.ui.tweezePositionsList.setCurrentRow(0)

    def sendPositionToActiveList(self, clicked):
        try:
            selectedItems = self.ui.tweezePositionsList.selectedItems()
            for item in selectedItems:
                self.ui.tweezePositionsList.takeItem(self.ui.tweezePositionsList.row(item))
                itemText = item.text()
                position = int(itemText.split(" ")[1])
                channelNo = int(itemText.split(" ")[3])
                self.ui.activePositionsList.addItem(itemText)
                self.activePositions.append((position, channelNo))
                self.tweezePositions.remove((position, channelNo))
                sys.stdout.write(f"Moved Pos: {position} and ChNo: {channelNo} to active list\n")
                sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"Moving channel to active list failed -- {e}\n")
            sys.stdout.flush()
        finally:
            if len(self.activePositions) > 0:
                self.ui.activePositionsList.setCurrentRow(0)


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

            # get 20 images from the last of the stack for display
            fileindices = [int(filename.stem)  for filename in list(directory.glob('*.tiff'))]
            fileindices.sort()

            if self.fetch_data['show20Images']:
                fileIndicesToGet = fileindices[-20:]
            else:
                fileIndicesToGet = fileindices
            
            files = [ directory / (str(i) + '.tiff') for i in fileIndicesToGet]

            number_images = len(files)
            if len(files) > 0:
                image = io.imread(files[0])
                for i in range(1, number_images):
                    image = np.concatenate((image, io.imread(files[i])), axis = 1)
                
                self.data = image

                sys.stdout.write(f"Image shape: {image.shape}\n")
                sys.stdout.flush()
            else:
                raise FileNotFoundError

        except Exception as e:
            sys.stdout.write(f"Data couldnot be fetched : {e}\n")
            sys.stdout.flush()
            self.data = np.random.normal(loc=0.0, scale=1.0, size=(100, 100))

        #self.data = np.random.normal(loc=0.0, scale=1.0, size=(100, 100))
        self.dataFetched.emit()

    def getData(self):
        return self.data
