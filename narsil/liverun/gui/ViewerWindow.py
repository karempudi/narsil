
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PySide6.QtCore import QFile, QIODevice, QTimer, Signal, Qt, QThread 
from PySide6.QtGui import QIntValidator
from PySide6.QtUiTools import QUiLoader

from narsil.liverun.ui.ui_ViewerWindow import Ui_ViewerWindow

from pathlib import Path
from skimage import io

import numpy as np
import sys
import pickle
import glob
import pyqtgraph as pg
import psycopg2 as pgdatabase

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
        self.databaseOk = False

        # thread object that will fetch images
        self.dataFetchThread = None
        self.dataThreadRunning = False

        self.currentPosition = None
        self.positionNoValidator = QIntValidator(0, 10000, self.ui.positionNoLine)
        self.channelNoValidator = QIntValidator(0, 200, self.ui.channelNoLine)
        self.currentChannelNo = None

        self.exptRunning = False

        self.showPhase = True
        self.showSeg = False

        self.show20Images = True
        self.showAllImages = False
        self.showProperties = False


        self.ui.activePositionsList.setSortingEnabled(True)
        self.ui.tweezePositionsList.setSortingEnabled(True)
        self.showActivePositions = True
        self.activePositions = []
        self.tweezePositions = []


        self.areaThreshold = None
        self.lengthThreshold = None
        self.noCellObjects = None

        self.sentPositions = []

        # filter parameters
        self.areaCutoff = None
        self.timepointCutoff = None
        self.numBlobsCutoff = None
        self.fraction = None

        self.setupButtonHandlers()
    
    def setSaveDir(self, saveDirPath):
        self.saveDir = Path(saveDirPath)

    def setDatabase(self, database):
        self.database = database
        self.databaseOk = True

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


        # area threshold slider handler
        self.ui.areaSlider.setMinimum(0)
        self.ui.areaSlider.setMaximum(6000)
        self.ui.areaSlider.valueChanged.connect(self.setAreaCutoff)

        # fraction slider handler
        self.ui.fractionSlider.setMinimum(0)
        self.ui.fractionSlider.setMaximum(100)
        self.ui.fractionSlider.valueChanged.connect(self.setFraction)

        # no of cell like objects slider
        self.ui.cellObjectsSlider.setMinimum(0)
        self.ui.cellObjectsSlider.setMaximum(20)
        self.ui.cellObjectsSlider.valueChanged.connect(self.setNumBlobsCutoff)

        # time point slider 
        self.ui.frameSlider.setMinimum(0)
        self.ui.frameSlider.setMaximum(100)
        self.ui.frameSlider.valueChanged.connect(self.setTimepointCutoff)

        # find all tweezalbe channels button
        self.ui.findLocationsButton.clicked.connect(self.getAllLocations)
        #self.ui.findLocationsButton.clicked.connect(self.getAllLocationsFake)


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

        # select if you want to get properties and plot at the same time you get images
        self.ui.plotPropertiesCheck.toggled.connect(self.setPlotPropertiesOption)

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

    def setPlotPropertiesOption(self, buttonState):
        self.showProperties = buttonState
        self.showCurrentPosition()
        sys.stdout.write(f"Plotting properies : {self.showProperties} from now on \n")
        sys.stdout.flush()

    def setViewActivePositions(self, buttonState):
        self.showActivePositions = buttonState
        self.showCurrentPosition()
        sys.stdout.write(f"Showing active positions: {self.showActivePositions}\n")
        sys.stdout.flush()

    def fetchData(self):
        # create a thread and call to get the data, once the data fetching
        # is done call the plotter
        #sys.stdout.write("Fetch Button clicked\n")
        #sys.stdout.flush()
        imgType = 'phase' if self.showPhase else 'cellSeg'
        numberOfImageToShow = True if self.show20Images else False

        if self.dataFetchThread is None:
            self.dataFetchThread =  dataFetchThread({'positionNo': self.currentPosition, 
                                'channelNo': self.currentChannelNo,
                                'type': imgType,
                                'dir': self.saveDir,
                                'show20Images': numberOfImageToShow,
                                'properties': ['area', 'lengths','objects'],
                                'database' : self.database,
                                'channelWidth': 36})
            self.dataFetchThread.start()
            self.dataFetchThread.dataFetched.connect(self.updateImage)

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

    def setAreaCutoff(self, value):
        self.areaCutoff = value
        sys.stdout.write(f"Area cutoff changed to: {value}\n")
        sys.stdout.flush()
    
    def setFraction(self, value):
        self.fraction = float(value) / 100.0
        sys.stdout.write(f"Fraction set to: {value/100.0}\n")
        sys.stdout.flush()

    def setTimepointCutoff(self, value):
        self.timepointCutoff = value
        sys.stdout.write(f"Timepoint cutoff: {value}\n")
        sys.stdout.flush()

    def setNumBlobsCutoff(self, value):
        self.numBlobsCutoff = value
        sys.stdout.write(f"Num of blobs cutoff: {value}\n")
        sys.stdout.flush()

    def getAllLocations(self, clicked):
        # loop through the database and create a list of positions based on filters 
        # that are set
        # check if all the filters are not None then fetch from database and apply filters 
        # and then generate list of active positions
        if self.databaseOk:
            sys.stdout.write(f"Grabbing all possible tweezable channels\n")
            sys.stdout.flush()

            self.activePositions.clear()
            self.ui.activePositionsList.clear()            

            try:
                con = None
                con = pgdatabase.connect(database=self.database['dbname'], user=self.database['dbuser'],
                                    password=self.database['dbpassword'])
                con.autocommit = True
                cur = con.cursor()
                cur.execute("SELECT * FROM growth")
                data = cur.fetchall()

                sorted_data = sorted(data, key=lambda element: element[0])

                formatted_data = [ [datapoint[0], datapoint[2], datapoint[3], datapoint[4], 
                                sum(pickle.loads(datapoint[5])), sum(pickle.loads(datapoint[6])), len(pickle.loads(datapoint[7]))]
                                for datapoint in sorted_data]
                
                data_numpy = np.asarray(formatted_data)
                num_positions = int(np.max(data_numpy[:, 1])) + 1
                max_channels = int(np.max(data_numpy[:, 3])) + 1

                for pos in range(num_positions):
                    for channel in range(max_channels):
                        pos_channel_data = data_numpy[np.argwhere(np.logical_and(data_numpy[:, 1] == pos, data_numpy[:, 3] == channel))]
                        #print(f"Pos: {pos} -- channelno: {channel} -- data: {len(pos_channel_data)}")
                        # sort the data by time and apply filters and select positions
                        pos_channel_data_list = pos_channel_data.squeeze(1).tolist()
                        channel_data = np.asarray(sorted(pos_channel_data_list, key=lambda element: element[2]))
                        
                        area_fraction = np.sum(channel_data[self.timepointCutoff:, 4] > self.areaCutoff)/ channel_data.shape[0]
                        blobs_fraction = np.sum(channel_data[self.timepointCutoff:, 6] > self.numBlobsCutoff)/ channel_data.shape[0]
                        
                        #print(blobs_fraction)
                        if area_fraction > self.fraction and blobs_fraction > self.fraction:
                            self.activePositions.append((pos, channel))
                            item = "Pos: " + str(pos) + " Ch: " + str(channel)
                            self.ui.activePositionsList.addItem(item)

                sys.stdout.write(f"Length of data : {len(formatted_data)}\n")
                sys.stdout.flush()

            
            except Exception as e:
                sys.stdout.write(f"Error in finding all tweezable positions -- {e}\n")
                sys.stdout.flush()
            
            finally:
                if con:
                    con.close()
        
        else:
            sys.stdout.write(f"Database not found\n")
            sys.stdout.flush()


    def getAllLocationsFake(self, clicked):
        # populate the list widget will all possible locations

        self.ui.activePositionsList.clear()
        self.activePositions.clear()
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


    def updateImage(self):
        #sys.stdout.write("Image received\n")
        #sys.stdout.flush()
        self.ui.imagePlot.setImage(self.dataFetchThread.getData()['image'], autoLevels=True, autoRange=False)
        self.ui.propertiesView.getPlotItem().plot(self.dataFetchThread.getData()['areas'], clear=True, pen='r')
        self.ui.propertiesView.getPlotItem().plot(self.dataFetchThread.getData()['lengths'], pen='b')
        self.ui.propertiesView.getPlotItem().plot(self.dataFetchThread.getData()['numobjects'], pen='g')
        #sys.stdout.write("Image plotted\n")
        #sys.stdout.flush()
        self.dataFetchThread.quit()
        self.dataFetchThread.wait()
        self.dataFetchThread = None
        return None
    

        

class dataFetchThread(QThread):

    dataFetched = Signal()

    def __init__(self, fetch_data):
        super(dataFetchThread, self).__init__()
        self.fetch_data = fetch_data
        self.data = None

    def run(self):
        # run code and fetch stuff
        sys.stdout.write(f"Image thread to get position: {self.fetch_data['positionNo']} and channelNo: {self.fetch_data['channelNo']} \n")
        #sys.stdout.write(f"{self.fetch_data['dir']}\n")
        #sys.stdout.flush()
        try:

            # Check the database and fetch the properties for plotting
            con = None
            con = pgdatabase.connect(database=self.fetch_data['database']['dbname'], user=self.fetch_data['database']['dbuser'],
                                password=self.fetch_data['database']['dbpassword'])
            con.autocommit=True
            cur = con.cursor()
            cur.execute("SELECT timepoint, areas, lengths, numobjects FROM growth WHERE position=%s AND channelno=%s", (self.fetch_data['positionNo'], self.fetch_data['channelNo']))

            data = cur.fetchall()
            sorted_data = sorted(data, key=lambda element: element[0])

            formatted_data = [ (datapoint[0], pickle.loads(datapoint[1]), pickle.loads(datapoint[2]), pickle.loads(datapoint[3]))
                                for datapoint in sorted_data]
            
            areas = np.zeros(shape=len(formatted_data))
            lengths = np.zeros(shape=len(formatted_data))
            numobjects = np.zeros(shape=len(formatted_data))
            for timepoint, datapoint in enumerate(formatted_data, 0):
                areas[timepoint] = sum(datapoint[1])
                lengths[timepoint] = sum(datapoint[2])
                numobjects[timepoint] = len(datapoint[3])
            # fetch all the images that are there in the directory
            # construct image by fetching

            cur.execute("SELECT locations FROM segment WHERE position=%s AND timepoint=%s", (self.fetch_data['positionNo'], 0))
            channelLocationsData = cur.fetchall()
            locations = pickle.loads(channelLocationsData[0][0])
  
            if self.fetch_data['type'] == 'phase':
                directory = self.fetch_data['dir'] / str(self.fetch_data['positionNo']) / "phaseFullImage"
            elif self.fetch_data['type'] == 'cellSeg':
                directory = self.fetch_data['dir'] / str(self.fetch_data['positionNo']) / "cellSegmentation" 

            # get 20 images from the last of the stack for display
            fileindices = [int(filename.stem)  for filename in list(directory.glob('*.tiff'))]
            fileindices.sort()

            sys.stdout.write(f"Directory -- {directory}\n")
            sys.stdout.flush()

            if self.fetch_data['show20Images']:
                fileIndicesToGet = fileindices[-20:]
            else:
                fileIndicesToGet = fileindices
            
            files = [ directory / (str(i) + '.tiff') for i in fileIndicesToGet]

            number_images = len(files)
            singleChannelLocation = locations[self.fetch_data['channelNo']]
            channelWidth = self.fetch_data['channelWidth'] // 2
            if len(files) > 0:

                image = io.imread(files[0])[:,
                            singleChannelLocation - channelWidth: singleChannelLocation + channelWidth]
                for i in range(1, number_images):
                    image_slice = io.imread(files[i])[:,
                            singleChannelLocation - channelWidth: singleChannelLocation + channelWidth]
                    image = np.concatenate((image, image_slice), axis = 1)
                
                self.data = {
                    'image': image,
                    'areas': areas,
                    'lengths': lengths,
                    'numobjects': numobjects
                }

                sys.stdout.write(f"Image shape: {image.shape}\n")
                sys.stdout.flush()
            else:
                raise FileNotFoundError

        except Exception as e:
            sys.stdout.write(f"Data couldnot be fetched : {e}\n")
            sys.stdout.flush()
            self.data =  {'image' : np.random.normal(loc=0.0, scale=1.0, size=(100, 100)),
                        'areas': np.random.normal(loc=0.0, scale=1.0, size=(40,)),
                        'lengths': np.random.normal(loc=0.0, scale=1.0, size=(40,)),
                        'numobjects': np.random.normal(loc=0.0, scale=1.0, size=(40,))
            }


        #self.data = np.random.normal(loc=0.0, scale=1.0, size=(100, 100))
        finally:
            if con:
                con.close()
            self.dataFetched.emit()


    def getData(self):
        return self.data
