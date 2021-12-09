# File containing classes that hooks up the buttons and processes behind
# the main ui classes generated from pyside6-uic
import os
import sys
import numpy as np
import time
import torch
import random
import json
from pathlib import Path
from collections import OrderedDict

from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PySide6.QtCore import QFile, QIODevice, QTimer, Signal, Qt, QThread
from PySide6.QtUiTools import QUiLoader

import pyqtgraph as pg
import psycopg2 as pgdatabase
from datetime import datetime
from skimage import io
from pycromanager import Bridge
from threading import Event

# utils and other imports from narsil
from narsil.liverun.utils import parsePositionsFile, getPositionList, padTo32
from narsil.liverun.utils import getPositionsMicroManager, getMicroManagerPresets, imgFilenameFromNumber

from narsil.segmentation.network import basicUnet, smallerUnet

# ui python classes import
from narsil.liverun.ui.ui_MainWindow import Ui_MainWindow
from narsil.liverun.exptDatabase import exptDatabase
from narsil.liverun.exptRun import exptRun, runProcesses

from narsil.liverun.gui.ExptSetupWindow import ExptSetupWindow
from narsil.liverun.gui.EventsWindow import EventsWindow
from narsil.liverun.gui.LiveWindow import LiveWindow
from narsil.liverun.gui.ViewerWindow import ViewerWindow

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
        self.tablesOk = False

        # setup button handlers
        self.setupButtonHandlers()

        # Create Subwindow objects and connect signal catchers
        self.setupWindow = ExptSetupWindow()
        self.setupWindow.setupDone.connect(self.receivedEvents)
        self.setupWindow.analysisSetupDone.connect(self.receivedAnalysisSetup)


        # viewer window to browse through different positions and checkout which 
        # positions to move to 
        self.viewerWindow = ViewerWindow()


        # initialize live window
        self.liveWindow = LiveWindow()


        # Create a database class object to be able to interact 
        # with the database when needed using a normal function
        # for most cases and threadpool when need to get larger data/files
        self.database = exptDatabase()
        self.databaseOk = False

        # starting process of the experiment
        self.exptRun = exptRun()
        self.exptRunStarted = False



    def setupButtonHandlers(self):
        # setup area buttons
        self.ui.setupButton.clicked.connect(self.showSetupWindow)
        # view setup button
        self.ui.viewExptSetupButton.clicked.connect(self.viewSetup)

        # write setup to file
        self.ui.writeSetupButton.clicked.connect(self.writeSetup)

        # load setup from file
        self.ui.loadSetupButton.clicked.connect(self.loadSetup)

        ############# controls button ############
        # create a database for the experiment
        self.ui.createDbButton.clicked.connect(self.createDatabase)
        # create the tables for analysis in the database
        self.ui.createTablesButton.clicked.connect(self.createTables)
        # delete database for the experiment
        self.ui.deleteDbButton.clicked.connect(self.deleteDatabase)
        # delete tables in the database
        self.ui.deleteTablesButton.clicked.connect(self.deleteTables)

        # start the experiment plots
        self.ui.startPlotsButton.clicked.connect(self.startPlotting)
        # stop the experiment plots
        self.ui.stopPlotsButton.clicked.connect(self.stopPlotting)
        # move to position no
        self.ui.moveToPositionButton.clicked.connect(self.moveToPosition)
        # live window can be used for tweezing
        self.ui.liveButton.clicked.connect(self.showLive)
        # tweeze positions
        self.ui.tweezePositionsButton.clicked.connect(self.showTweezablePositions)

        ############ viewer button ###############

        ############ statistics button ###########
        self.ui.deadAliveStatsButton.clicked.connect(self.showViewer)

    def showViewer(self):
        self.viewerWindow.show()

    # signal catcher from setup window
    def receivedEvents(self, exptSettings):
        self.exptSetupSettings = exptSettings
        self.exptSetupOk = True
    
    # signal catcher from setup window
    def receivedAnalysisSetup(self, analysisSettings):
        self.analysisSetupSettings = analysisSettings
        self.analysisSetupOk = True

        
        self.viewerWindow.setSaveDir(self.analysisSetupSettings['saveDir'])

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

    # write the setup file to the specified location
    # Setup file should have everything that is needed for the exptRun class, that 
    # runs all the analysis functions
    def writeSetup(self):
        writeDict = {}
        if self.exptSetupOk:
            writeDict['setup'] = self.exptSetupSettings
        
        if self.analysisSetupOk:
            writeDict['analysis'] = self.analysisSetupSettings

        if self.databaseOk:
            writeDict['database'] = {
                'dbname': self.database.dbname,
                'dbuser': 'postgres',
                'dbpassword': 'postgres',
                'tables': self.database.tables
            }
        
        saveDir = QFileDialog.getExistingDirectory(self,
                self.tr("Save expt setup file"), '.', 
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
                )
        #filename = self.exptSetupSettings['exptNo'].lower() + '.json'
        filename = 'exp21bp000' + '.json'
        saveFilename = Path(saveDir) / filename

        with open(saveFilename, 'w') as filehandle:
            json.dump(writeDict, filehandle)

        sys.stdout.write(f"{saveFilename} -- Expt Setup written\n")
        sys.stdout.flush()


    # Load setup from file,
    # in case the UI dies, we will reload the setup from the file and run as usual
    def loadSetup(self):
        # load the file using a file dialog and set everything that is needed for all the windows to work

        filename = QFileDialog.getOpenFileName(self,
                    self.tr("Open a Expt setup file"), '.', self.tr("Expt setup file (*.json)"))
        
        if filename == '':
            msg = QMessageBox()
            msg.setText("Expt setup file not selected")
            msg.exec()
        else:
            try:
                with open(Path(filename[0])) as json_file:
                    exptDict = json.load(json_file)

                if 'setup' in exptDict:
                    self.exptSetupSettings = exptDict['setup']
                    self.exptSetupOk = True
                else:
                    sys.stdout.write(f"Expt Setup not found\n")
                    sys.stdout.flush()

                if 'analysis' in exptDict:
                    self.analysisSetupSettings = exptDict['analysis']
                    self.analysisSetupOk = True
                    self.viewerWindow.setSaveDir(self.analysisSetupSettings['saveDir'])

                else:
                    sys.stdout.write(f"Analysis Setup not found\n")
                    sys.stdout.flush()
                
                if 'database' in exptDict:
                    self.database.dbname = self.exptSetupSettings['exptNo'].lower()
                    self.database.tables = exptDict['database']['tables']
                    self.databaseOk = True
                    self.viewerWindow.setDatabase(exptDict['database'])
                else:
                    sys.stdout.write(f"Database setup not found\n")
                    sys.stdout.flush()

            except Exception as e:
                sys.stdout.write(f"Error in loading setup -- {e}\n")
                sys.stdout.flush()

    
    ############ controls button handlers ##################
    def createDatabase(self):
        if self.exptSetupOk:
            self.database.dbname = self.exptSetupSettings['exptNo'].lower()
            self.database.createDatabase()
            self.databaseOk = True

            self.viewerWindow.setDatabase({'dbname': self.database.dbname,
                                         'dbuser': 'postgres',
                                         'dbpassword': 'postgres',
                                         'tables': self.database.tables
                                        })

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

    def startPlotting(self):
            self.initializePlots()

            # timer that update plots every few seconds
            self.timer = QTimer()
            self.timer.setInterval(1000)
            self.timer.timeout.connect(self.updateStatusPlots)
            self.timer.start()

    def stopPlotting(self):
        self.timer.stop()

    def moveToPosition(self):
        pass
    
    def showLive(self):
        self.liveWindow.setParameters(self.analysisSetupSettings)
        self.liveWindow.show()
    
    def showTweezablePositions(self):
        pass

    ############ viewer button handlers ####################


    ############ statistics button handlers ################

    ############ Plotting handlers #########################

    # initialize the plots before starting the experiment
    def initializePlots(self):

        # arrival plot
        if 'arrival' in self.database.tables:
            self.acquiredPlot = self.ui.imgAcquiredPlot.getPlotItem()
            self.acquiredPlot.clear()
            self.acquiredPlot.setLabel('left', text='timepoint')
            self.acquiredPlot.setLabel('bottom', text='position')
            self.acquiredPlot.setTitle(title='Images Acquired')
        
        if 'segment' in self.database.tables:
            self.segmentPlot = self.ui.imgSegPlot.getPlotItem()
            self.segmentPlot.clear()
            self.segmentPlot.setLabel('left', text='timepoint')
            self.segmentPlot.setLabel('bottom', text='position')
            self.segmentPlot.setTitle(title='Images segmented')
        

    # plots are update every few seconds, this timer can be set
    # in the intialization of the UI
    def updateStatusPlots(self):
        # Get data from each table and update the plot data

        sys.stdout.write(f"Refreshing plots: {datetime.now()}\n")
        sys.stdout.flush()
        acquiredData = self.database.queryDataForPlots(tableName='arrival')
        segmentData = self.database.queryDataForPlots(tableName='segment')
        self.acquiredPlot.plot(np.array(acquiredData), symbol='o', pen=pg.mkPen(None))
        self.segmentPlot.plot(np.array(segmentData), symbol='o', pen=pg.mkPen(None))


    ############ other miscalleneous handlers ##############


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

