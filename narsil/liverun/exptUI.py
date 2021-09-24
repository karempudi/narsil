# File containing classes that hooks up the buttons and processes behind
# the main ui classes generated from pyside6-uic
import os
import sys
import numpy as np
from pathlib import Path
from collections import OrderedDict

from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PySide6.QtCore import QFile, QIODevice, QTimer, Signal, Qt
from PySide6.QtUiTools import QUiLoader

import pyqtgraph as pg
import psycopg2 as pgdatabase
from datetime import datetime

# utils and other imports from narsil
from narsil.liverun.utils import parsePositionsFile, getPositionList
from narsil.liverun.utils import getPositionsMicroManager, getMicroManagerPresets


# ui python classes import
from narsil.liverun.ui.ui_MainWindow import Ui_MainWindow
from narsil.liverun.ui.ui_EventsWindow import Ui_EventsWindow
from narsil.liverun.ui.ui_SetupWindow import Ui_SetupWindow
from narsil.liverun.exptDatabase import exptDatabase
from narsil.liverun.exptRun import exptRun

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Narsil Live Expt Analysis")

        # expt and analysis objects
        self.exptSetupSettings = None
        self.analysisSetupSettings = None
        self.exptSetupOk = False
        self.analysisSetupOk = False

        # database, tables Ok
        self.databaseOk = False
        self.tablesOk = False

        # setup button handlers
        self.setupButtonHandlers()

        # Create Subwindow objects and connect signal catchers
        self.setupWindow = ExptSetupWindow()
        self.setupWindow.setupDone.connect(self.receivedEvents)
        self.setupWindow.analysisSetupDone.connect(self.receivedAnalysisSetup)

        # Create a database class object to be able to interact 
        # with the database when needed using a normal function
        # for most cases and threadpool when need to get larger data/files
        self.database = exptDatabase()
        self.databaseOk = False

        # starting process of the experiment
        self.exptRun = exptRun()
        self.exptRunStarted = False

        # timer that update plots every few seconds
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.updateStatusPlots)
        self.timer.start()



    def setupButtonHandlers(self):
        # setup area buttons
        self.ui.setupButton.clicked.connect(self.showSetupWindow)
        # view setup button
        self.ui.viewExptSetupButton.clicked.connect(self.viewSetup)

        ############# controls button ############
        # create a database for the experiment
        self.ui.createDbButton.clicked.connect(self.createDatabase)
        # create the tables for analysis in the database
        self.ui.createTablesButton.clicked.connect(self.createTables)
        # delete database for the experiment
        self.ui.deleteDbButton.clicked.connect(self.deleteDatabase)
        # delete tables in the database
        self.ui.deleteTablesButton.clicked.connect(self.deleteTables)

        # start the experiment
        self.ui.startExptButton.clicked.connect(self.startExpt)
        # stop the experiment
        self.ui.stopExptButton.clicked.connect(self.stopExpt)
        # move to position no
        self.ui.moveToPositionButton.clicked.connect(self.moveToPosition)
        # live window can be used for tweezing
        self.ui.liveButton.clicked.connect(self.showLive)
        # tweeze positions
        self.ui.tweezePositionsButton.clicked.connect(self.showTweezablePositions)

        ############ viewer button ###############

        ############ statistics button ###########

    # signal catcher from setup window
    def receivedEvents(self, exptSettings):
        self.exptSetupSettings = exptSettings
        self.exptSetupOk = True
    
    # signal catcher from setup window
    def receivedAnalysisSetup(self, analysisSettings):
        self.analysisSetupSettings = analysisSettings
        self.analysisSetupOk = True

    #############  setup button handlers ##################

    # setup button handler
    def showSetupWindow(self):
        self.setupWindow.show()
    
    # view setup button handler
    def viewSetup(self):
        # calculate what to display the window
        expt_string = ""
        for key, values in self.exptSetupSettings.items():
            if key == "events":
                expt_string += str(key) + " no : " + str(len(values))
                expt_string += "\n"
            else:
                expt_string += str(key) + " : " + str(values)
                expt_string += "\n"
        # construct a message box on the fly and 

        analysis_string = ""
        for key, values in self.analysisSetupSettings.items():
            analysis_string += str(key) + " : " + str(values)
            analysis_string += "\n"

        msg = QMessageBox()
        msg.setWindowTitle("Setup and Analysis Settings")
        msg.setText(expt_string + "\n\n" + analysis_string)
        msg.exec()     
    
    ############ controls button handlers ##################
    def createDatabase(self):
        if self.exptSetupOk:
            self.database.dbname = self.exptSetupSettings['exptNo'].lower()
            self.database.createDatabase()
            self.databaseOk = True
    
    def createTables(self):
        # depending on what you analyze you can create appropriate table
        if self.analysisSetupOk == True:
            # make a list of tables to constrct and construct them 
            tables = []
            tables.extend(['arrival', 'segment'])
            if self.analysisSetupSettings['deadAlive']:
                tables.append('deadalive')
            
            if self.analysisSetupSettings['growthRates']:
                tables.append('growth')
            
            self.database.tables = tables
            print(self.database.tables)
            self.database.createTables()

    def deleteDatabase(self):
        self.database.deleteDatabase()
        self.databaseOk = False
    
    def deleteTables(self):
        self.database.deleteTables()
        self.databaseOk = False

    def startExpt(self):
        # basically launch a QProcess and call scirpt with appropriate arugments
        if self.exptSetupOk:
            print("Expt setup is ok .. Running now ...")
            self.exptRun.acquireEvents = self.exptSetupSettings['events'] 
            self.exptRun.run()

    def stopExpt(self):
        # send kill signal to the QProcess running the experiment
        if self.exptSetupOk:
            self.exptRun.stop()
   
    def moveToPosition(self):
        pass
    
    def showLive(self):
        pass
    
    def showTweezablePositions(self):
        pass

    ############ viewer button handlers ####################


    ############ statistics button handlers ################


    ############ other miscalleneous handlers ##############

    def updateStatusPlots(self):
        pass
    

class ExptSetupWindow(QMainWindow):

    # Emit when setup is done and handled in the
    # parent window
    setupDone = Signal(dict)

    # analysis setup done, emit when setup is done
    analysisSetupDone = Signal(dict)

    def __init__(self):
        super(ExptSetupWindow, self).__init__()
        self.ui = Ui_SetupWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Experiment Setup Window")

        # validataion Variables
        # Check names, positions reads, events and presets are all
        # correctly setup
        self.exptSettingsValidated = False
        # Check if all the algorithms are correctly marked,
        # All database tables are connected, created and ready to go
        self.analysisSettingsValidated = False

        # activate button handlers
        self.setupButtonHandlers()

        # Data that is needed tobe stored
        self.exptNo = None
        self.positionsFromFile = True
        self.positionsFileName = None # positions filename 
        self.eventsCreated = False # Flag to know if events were created
        self.exptSetupData = None # assign later when the events windows closes
        self.exptDir = '.'
        self.exptSettings= {}

        # additional window references that are needed
        self.eventsWindow = EventsWindow()
        self.eventsWindow.eventsCreated.connect(self.receivedEvents)

        # analysis setting that need to be stored
        self.analysisSettings = {"cellNet": "normal", 
                    "channelSeg": None, "deadAlive": None, "growthRates": None,
                    "cellSegNetModelPath": None, "channelSegNetModelPath": None,
                    "imageHeight": None, "imageWidth": None
                    }

    
    # Receiving events list from the create Events subwindow
    def receivedEvents(self, sentdata):
        self.exptSetupData = sentdata
        self.eventsCreated = True
        #print(self.events)
        print("Events received .... ")
        print("Events in the setup window are set ..")
    
    def setupButtonHandlers(self):
        
        #########################################################
        ############## Expt Setup Buttons #######################
        #########################################################
        # expt No set button
        self.ui.exptNoSetButton.clicked.connect(self.setExptNo)
        # clear expt No button
        self.ui.exptNoClearButton.clicked.connect(self.clearExptNo)
        # click the get positions radio buttons
        # TODO: add stuff to get how you are going to get positions
        self.ui.fromFile.toggled.connect(self.fileOptionClicked)
        self.ui.fromMicroManager.toggled.connect(self.micromanagerOptionClicked)

        # select file button
        self.ui.fileSelectionButton.clicked.connect(self.selectPositionsFile)

        # create events button, opens new window
        self.ui.eventsCreationButton.clicked.connect(self.createEvents)

        # validate Expt setup button
        self.ui.validateExptSetupButton.clicked.connect(self.validateExptSetup)

        #########################################################
        ############## Analysis Setup Buttons ###################
        #########################################################
        # detect changes in net-type selection
        self.ui.selectNet.currentIndexChanged.connect(self.changeNetType)
        # select cell segment net model file
        self.ui.cellSegNetFilePathButton.clicked.connect(self.setCellSegModelPath)
        # select channel segmet net model file
        self.ui.channelSegNetFilePathButton.clicked.connect(self.setChannelSegModelPath)
        # image height set
        self.ui.imageHeight.textChanged.connect(self.setImageHeight)
        # image width set
        self.ui.imageWidth.textChanged.connect(self.setImageWidth)
        # checkbox for channel segmentation
        self.ui.segChannels.stateChanged.connect(self.setChannelSegmentation)
        # deadAlive for rudimentary tracking
        self.ui.calcDeadAlive.stateChanged.connect(self.setDeadAlive)
        # Growth Rates for full blown analysis
        self.ui.calcGrowthRates.stateChanged.connect(self.setGrowthRates)
        # validate analysis setup
        self.ui.validateAnalysisSetupButton.clicked.connect(self.validateAnalysisSetup)



        # save button and the close button
        self.ui.closeExptSetupButton.clicked.connect(self.closeWindow)
        

        
    def setExptNo(self, clicked):
        if self.exptNo != None and (self.exptNo != self.ui.exptNoText.text()):
            dlg = QMessageBox()
            dlg.setWindowTitle("Please confirm!!!")
            dlg.setText(f"""Experiment no already set to {self.exptNo}.
                            Change to {self.ui.exptNoText.text()}""")
            dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            dlg.setIcon(QMessageBox.Question)
            button = dlg.exec()

            if button == QMessageBox.Yes:
                self.exptNo = self.ui.exptNoText.text()
            elif button == QMessageBox.No:
                self.ui.exptNoText.setText(self.exptNo)
        else:
            self.exptNo = self.ui.exptNoText.text()

        print(f"Experiment no set to {self.exptNo}")
    
    def clearExptNo(self, clicked):
        self.exptNo = None
        self.ui.exptNoText.setText("EXP21BP000")

    def fileOptionClicked(self, clicked):
        self.positionsFromFile = self.ui.fromFile.isChecked()
        self.eventsWindow.usePositionsFile = self.positionsFromFile 
        print(f"File option toggled {self.positionsFromFile}")


    def micromanagerOptionClicked(self, clicked):
        self.positionsFromFile = not self.ui.fromMicroManager.isChecked()
        self.eventsWindow.usePositionsFile = self.positionsFromFile
        print(f"Micromanger option toggled: {self.positionsFromFile}")

    def selectPositionsFile(self, clicked):
        if self.positionsFromFile == True:
            filename = QFileDialog.getOpenFileName(self, 
                self.tr("Open position file"), self.exptDir , self.tr("Position files (*.pos)"))

            self.positionsFileName = filename[0]
            print(filename)
            print(f"Positions file set to {self.positionsFileName}")

            # set positions filename so that positions can be picked up in
            # that window
            self.eventsWindow.positionsFileName = self.positionsFileName
            if self.positionsFileName == '':
                self.positionsFileName = None
                msg = QMessageBox()
                msg.setText("Positions file not set")
                msg.exec()
        else:
            msg = QMessageBox()
            msg.setText("Positions are coming from Micromanager directly")
            msg.exec()
        
    # Window gets created in the initialization
    # you just show it
    def createEvents(self, clicked):
        self.eventsWindow.show()
    
    def validateExptSetup(self, clicked):
        # Open a window and then ask a question to accept or not  
        if self.exptNo == None:
            msg = QMessageBox()
            msg.setText("Expt number not set")
            msg.setIcon(QMessageBox.Warning)
            msg.exec()
            return
        
        if self.positionsFromFile == True and self.positionsFileName == None:
            msg = QMessageBox()
            msg.setText("Positions File name or micromanager option is not set")
            msg.setIcon(QMessageBox.Critical)
            msg.exec()
            return
        
        # TODO: checking for micromanager option is done yet
        if self.positionsFromFile == False and self.eventsCreated == False:
            msg = QMessageBox()
            msg.setText("Micromanager events are not created ..")
            msg.setIcon(QMessageBox.Critical)
            msg.exec()
            return

        if self.eventsCreated == False or self.exptSetupData == None:
            msg = QMessageBox()
            msg.setText("Events not created")
            msg.setIcon(QMessageBox.Critical)
            msg.exec()
            return
        
        self.exptSettingsValidated = True
        # construct everything about experiment setup in a dictionary
        # to be passsed around if needed in other places

        self.exptSettings = {
            "events": self.exptSetupData['events'],
            "nPositions": self.exptSetupData['nPositions'],
            "slowPositions": self.exptSetupData['slowPositions'],
            "fastPositions": self.exptSetupData['fastPositions'],
            "nTimePoints": self.exptSetupData['nTimePoints'],
            "timeInterval": self.exptSetupData['timeInterval'],
            "exptNo": self.exptNo,
            "positionsFileName": self.positionsFileName,
            "posFromFile": self.positionsFromFile
        }
        self.setupDone.emit(self.exptSettings)
    
        msg = QMessageBox()
        msg.setText("Experiment settings validated")
        msg.setIcon(QMessageBox.Information)
        msg.exec()


    def changeNetType(self, i):
        currentType = self.ui.selectNet.currentText()
        self.analysisSettings["cellNet"] = str(currentType)

    def setCellSegModelPath(self):
        filename = QFileDialog.getOpenFileName(self, 
                self.tr("Open Cell Segment Model Path"), '.', self.tr("Pytorch model files (*.pth)"))

        if filename == '':
            msg = QMessageBox()
            msg.setText("Cell Segmentation model not set")
            msg.exec()
        else:
            self.analysisSettings["cellSegNetModelPath"] = filename[0]
            print(f"Cell Seg model file: {self.analysisSettings['cellSegNetModelPath']}")


    def setChannelSegModelPath(self):
        filename = QFileDialog.getOpenFileName(self, 
                self.tr("Open Channel Segment Model Path"), '.', self.tr("Pytorch model files (*.pth)"))

        if filename == '':
            msg = QMessageBox()
            msg.setText("Channel Segmentation model not set")
            msg.exec()
        else:
            self.analysisSettings["channelSegNetModelPath"] = filename[0]
            print(f"Channel Seg model file: {self.analysisSettings['channelSegNetModelPath']}")

    def setImageHeight(self):
        height = self.ui.imageHeight.text()
        # set to the nearest multiple of 16
        try:
            intHeight = int(height)
            if intHeight%32 != 0:
                intHeight = intHeight - intHeight%32 + 32
            self.analysisSettings['imageHeight'] = intHeight
        except:
            msg = QMessageBox()
            msg.setText("Height should be a number")
            msg.exec()

    def setImageWidth(self):
        width = self.ui.imageWidth.text()
        # set to the nearest multiple of 16
        try:
            intWidth = int(width)
            if intWidth%32 != 0:
                intWidth = intWidth - intWidth%32 + 32
            self.analysisSettings['imageWidth'] = intWidth
        except:
            msg = QMessageBox()
            msg.setText("Weight should be a number")
            msg.exec()

    def setChannelSegmentation(self, s):
        self.analysisSettingsValidated = False
        if s == Qt.Checked:
            self.analysisSettings["channelSeg"] = True
        elif s == Qt.Unchecked:
            self.analysisSettings["channelSeg"] = None


    def setDeadAlive(self, s):
        self.analysisSettingsValidated = False
        if s == Qt.Checked:
            self.analysisSettings["deadAlive"] = True
        elif s == Qt.Unchecked:
            self.analysisSettings["deadAlive"] = None
    
    def setGrowthRates(self, s):
        self.analysisSettingsValidated = False
        if s == Qt.Checked:
            self.analysisSettings["growthRates"] = True
        elif s == Qt.Unchecked:
            self.analysisSettings["growthRates"] = None
    
    def validateAnalysisSetup(self, clicked):
        if self.exptSettingsValidated != True:
            msg = QMessageBox()
            msg.setText("Experimental setup not completed !!! Do it first")
            msg.setIcon(QMessageBox.Warning)
            msg.exec()
            return
        
        if self.analysisSettings["channelSeg"] == None:
            msg = QMessageBox()
            msg.setText("Channel Segmentation will not be done ...")
            msg.setIcon(QMessageBox.Information)
            msg.exec()
        
        if self.analysisSettings["deadAlive"] == None:
            msg = QMessageBox()
            msg.setText("Dead Alive calcultations will not be done ...")
            msg.setIcon(QMessageBox.Information)
            msg.exec()
        
        if self.analysisSettings["growthRates"] == None:
            msg = QMessageBox()
            msg.setText("Growth Rates calcultations will not be done ...")
            msg.setIcon(QMessageBox.Information)
            msg.exec()

        self.analysisSettingsValidated = True
        self.analysisSetupDone.emit(self.analysisSettings)
        
        msg = QMessageBox()
        msg.setText("Analysis settings done")
        msg.setIcon(QMessageBox.Information)
        msg.exec()
    
    # close the window, just the view, the object still exists
    def closeWindow(self):
        self.close()



class EventsWindow(QMainWindow):

    eventsCreated = Signal(dict)

    def __init__(self):
        super(EventsWindow, self).__init__()
        self.ui = Ui_EventsWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Events Creation Window")


        # this value is set when the user successfully selects
        # the a correct positions file
        self.usePositionsFile = None
        self.positionsFileName = None

        # will set true if the positions are parsed correctly
        # These positions will by default be loaded into fast postions side
        self.parsedPositions = None

        # enable sorting in the lists of positions
        self.ui.fastPositions.setSortingEnabled(True)
        self.ui.slowPositions.setSortingEnabled(True)

        # set button handlers
        self.setupButtonHandlers()

        # set channel presets
        availablePresets = getMicroManagerPresets()
        for preset in availablePresets:
            self.ui.presets.addItem(preset)

        # Minimum time to wait before you go back to acquire the first position
        self.minTimeInterval = 0
        # attach the time interval spinner to the value
        self.ui.minTimeIntervalSpinBox.valueChanged.connect(self.timeIntervalChanged)

        # number of time points 
        self.nTimePoints = 1
        # attach the time point spinner to the value
        self.ui.nTimePointsSpinBox.valueChanged.connect(self.nTimePointsChanged)

        self.finalizedPositions = False
        self.listFastPositions = []
        self.listSlowPositions = []
        self.finalEvents = []
        self.finalizedEvents = False
        self.sentData = {}


    def setupButtonHandlers(self):
        # By default get positions will fill in all the positions in Fast
        self.ui.getPositionsButton.clicked.connect(self.setFastPositionsDefault)

        # Finalize positions will get final positions for both slow and fast
        self.ui.finalizePositionsButton.clicked.connect(self.setFinalPositions)

        # Reset all positions and clean up
        self.ui.resetPositionsButton.clicked.connect(self.resetAllPositions)

        # move from fast to slow
        self.ui.sendToSlowButton.clicked.connect(self.moveToSlow)

        # move from slow to fast
        self.ui.sendToFastButton.clicked.connect(self.moveToFast)

        # Add channel and exposure times to the list 
        self.ui.addPresetButton.clicked.connect(self.addPresetToList)

        # Remove the selected channels from the list
        self.ui.removePresetButton.clicked.connect(self.removePresetFromList)

        # Construct events in the correct format
        self.ui.constructEventsButton.clicked.connect(self.constructFinalEvents)

        # Reset events
        self.ui.resetEventsButton.clicked.connect(self.resetFinalEvents)

        # Close window and clean up correctly
        self.ui.closeWindowButton.clicked.connect(self.closeWindow)

    def addPresetToList(self, clicked):
        currentPreset = self.ui.presets.currentText()
        currentExposure = self.ui.exposure.text()
        try:
            intExposure = int(currentExposure)
            self.ui.channelExposureList.addItem(currentPreset + " " + str(intExposure) + " ms")
        except: 
            msgBox = QMessageBox()
            msgBox.setText("Exposure should be a number")
            msgBox.exec()


    def removePresetFromList(self, clicked):
        selectedRow = self.ui.channelExposureList.currentRow()
        selectedItem = self.ui.channelExposureList.takeItem(selectedRow)
        del selectedItem

    def timeIntervalChanged(self, timeIntervalValue):
        self.minTimeInterval = timeIntervalValue
    
    def nTimePointsChanged(self, nTimePoints):
        if nTimePoints >= 1:
            self.nTimePoints = nTimePoints
        else:
            self.ui.nTimePointsSpinBox.setValue(1)

    def constructFinalEvents(self, clicked):

        if not self.finalizedPositions:
            msgBox = QMessageBox()
            msgBox.setText("Positions not finalized. Add them and click Finalize Positions ..")
            msgBox.exec()
            return

        nPositions = len(self.listFastPositions) + len(self.listSlowPositions)

        allPositions = self.listFastPositions + self.listSlowPositions

        allPositions.sort(key = lambda x: int(x[3:]))
        #print(allPositions)
        nPresets = self.ui.channelExposureList.count()


        if nPresets == 0:
            msgBox = QMessageBox()
            msgBox.setText("Presets are not set.. So, events are not created")
            msgBox.exec()
            return

        presetExposureDict = OrderedDict()
        for i in range(nPresets):
            presetAndExposure = self.ui.channelExposureList.item(i).text()
            preset = presetAndExposure.split(" ")[0]
            exposure = int(presetAndExposure.split(" ")[1])
            presetExposureDict[preset] = exposure

        print(presetExposureDict)
        if list(presetExposureDict.items())[0][0] != "phase":
            dlg = QMessageBox()
            dlg.setWindowTitle("Please confirm !!!")
            dlg.setText(f"First preset is not phase .. Fast and slow movements will be diabled by default")
            dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            dlg.setIcon(QMessageBox.Question)
            button = dlg.exec()

            if button == QMessageBox.No:
                msg = QMessageBox()
                msg.setText("Event creation terminated")
                msg.exec()


        # for each position construct the events based on the channels
        # added in the presets
        for timePoint in range(self.nTimePoints):
            for i, position in enumerate(self.positionsData, 0):
                # iterate through the presets as well and add them
                for preset, exposure in presetExposureDict.items():
                    event = {}
                    event['axes'] = {'time': timePoint, 'position': i}
                    event['min_start_time'] = 0 + (timePoint) * self.minTimeInterval
                    event['x'] = self.positionsData[position]['x_coordinate']
                    event['y'] = self.positionsData[position]['y_coordinate']
                    event['z'] = self.positionsData[position]['pfs_offset']

                    # check if position is fast or slow and
                    # add appropriate presets
                    if position in self.listSlowPositions:
                        if preset == "phase":
                            #print(f"{position} is slow so adding slow presets")
                            event['channel'] = {'group': 'image', 'config': preset + "Slow"}
                        else:
                            event['channel'] = {'group': 'image', 'config': preset}

                    else:
                        if preset == "phase":
                            #print(f"{position} is fast so adding fast presets")
                            event['channel'] = {'group': 'image', 'config': preset + "Fast"}
                        else:
                            event['channel'] = {'group': 'image', 'config': preset}
                    event['exposure'] = exposure
                    print(event)
                    self.finalEvents.append(event)
                    print("-----")
        # Events have been created
        self.finalizedEvents = True
        self.sentData['events'] = self.finalEvents
        self.sentData['nTimePoints'] = self.nTimePoints
        self.sentData['nPositions'] = len(self.positionsData.keys())
        self.sentData['slowPositions'] = self.listSlowPositions
        self.sentData['fastPositions'] = self.listFastPositions
        self.sentData['timeInterval'] = self.minTimeInterval

        self.eventsCreated.emit(self.sentData)

    def resetFinalEvents(self, clicked):
        self.finalEvents = []
        self.finalizedEvents = False
        msg = QMessageBox()
        msg.setText("Events have been reset .. Set Events again")
        msg.exec()

    def closeWindow(self, clicked):
        # This will close the window
        self.close()
    
    def setFastPositionsDefault(self, clicked):

        # check if positions/micromanager file is not set
        # then do some default positions for debugging the UI
        if self.usePositionsFile is None:
            print(f"Positions from file: {self.usePositionsFile}")
            self.positionsData = getPositionList(None)
            msg = QMessageBox()
            msg.setText("Plese check the file option in Expt Setup Window")
            msg.setIcon(QMessageBox.Information)
            msg.exec()
            return
        # use the positoins file
        elif self.usePositionsFile == True:
            if self.positionsFileName == '' or self.positionsFileName is None:
                msg = QMessageBox()
                msg.setText("Select the positions file in the Expt Setup Window")
                msg.setIcon(QMessageBox.Warning)
                msg.exec()
            print("Sending positions to parser .... ")
            print(f"Positions from file: {self.usePositionsFile}")
            self.positionsData = getPositionList(self.positionsFileName)
        
        elif self.usePositionsFile == False:
            print(f"Positions from file: {self.usePositionsFile}")
            self.positionsData = getPositionsMicroManager()

        for position in self.positionsData:
            self.ui.fastPositions.addItem(position)
    
    def setFinalPositions(self, clicked):
        if not self.finalizedPositions:
            nFastPositions = self.ui.fastPositions.count()
            nSlowPositions = self.ui.slowPositions.count()

            for i in range(nFastPositions):
                self.listFastPositions.append(self.ui.fastPositions.item(i).text())
            
            for i in range(nSlowPositions):
                self.listSlowPositions.append(self.ui.slowPositions.item(i).text())

            print("Final positions set .... ")
            print(self.listSlowPositions)
            print("---------------")
            print(self.listFastPositions)

            # set that you have finalized positions previously
            self.finalizedPositions = True

        else:
            msgBox = QMessageBox()
            msgBox.setText("You finalized positions previously.. Try resettting positons.")
            msgBox.exec()


    def resetAllPositions(self, clicked):
        self.ui.fastPositions.clear()
        self.ui.slowPositions.clear()
        self.finalizedPositions = False
        self.listFastPositions = []
        self.listSlowPositions = []
        msgBox = QMessageBox()
        msgBox.setText("All positions cleared. Reload positions")
        msgBox.exec()

    def moveToFast(self, clicked):
        selectedRow = self.ui.slowPositions.currentRow()
        selectedPosition = self.ui.slowPositions.takeItem(selectedRow)
        self.ui.fastPositions.addItem(selectedPosition)

    def moveToSlow(self, clicked):
        selectedRow = self.ui.fastPositions.currentRow()
        selectedPosition = self.ui.fastPositions.takeItem(selectedRow)
        self.ui.slowPositions.addItem(selectedPosition)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

