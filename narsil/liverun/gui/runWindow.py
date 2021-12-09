from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PySide6.QtCore import QFile, QIODevice, QTimer, Signal, Qt, QThread 
from PySide6.QtGui import QIntValidator
from PySide6.QtUiTools import QUiLoader

from narsil.liverun.ui.ui_RunWindow import Ui_runWindow
from narsil.liverun.exptRun import exptRun, runProcesses

from pathlib import Path
from skimage import io

import numpy as np
import sys
import glob
import json
import concurrent.futures


class RunWindow(QMainWindow):

    def __init__(self):
        super(RunWindow, self).__init__()
        self.ui = Ui_runWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Experiment Run window")


        self.exptRunDict = None

        self.exptRun = exptRun()
        self.exptRunStarted = False

        self.setupOk = False

        self.filesAllocated = False

        self.setupButtonHandlers()
    
    def setupButtonHandlers(self):
        
        # load the data from the experiment file
        self.ui.loadButton.clicked.connect(self.loadExptRun)

        # allocate files on disk before starting the experiment, especially the small files
        self.ui.makeFilesButton.clicked.connect(self.allocateFiles)

        # Run the experiment
        self.ui.runButton.clicked.connect(self.runExpt)

        self.ui.stopButton.clicked.connect(self.stopExpt)
    
    def allocateFiles(self):

        if not self.setupOk:
            sys.stdout.write(f"Setup is not done correctly. Check all setup variables are set ... \n")
            sys.stdout.flush()

        if not self.filesAllocated:
            timepoints = self.exptRunDict['setup']['nTimePoints']
            nPositions = self.exptRunDict['nPositions']
            saveDir = self.exptRunDict['analysis']['saveDir']
            # pull this from channel process parameters later set as constant for now
            channelWidth = 36
            channelHeight = self.exptRunDict['analysis']['imageHeight']

            # loop over the positions and concurrently create all the file needed 


            sys.stdout.write(f"Pre-Allocating files on disk ...\n")
            sys.stdout.flush()
            self.filesAllocated = True
        else:
            sys.stdout.write(f"Files already allocated ...\n")
            sys.stdout.flush()

    def loadExptRun(self, clicked):
        try:
            filename = QFileDialog.getOpenFileName(self,
                    self.tr("Open an experiment setup json file"), '.',
                    self.tr("Expt setup json file (*.json)"))
            
            if filename == '':
                msg = QMessageBox()
                msg.setText("Expt setup file not selected")
                msg.exec()
            else:
                with open(Path(filename[0])) as json_file:
                    self.exptRunDict = json.load(json_file)
           
        except Exception as e:
            sys.stdout.write(f"Error in loading the experimental setup file -- {e}\n")
            sys.stdout.flush()
        finally:

            if ('setup' in self.exptRunDict) and ('analysis' in self.exptRunDict) and ('database' in self.exptRunDict):
                self.setupOk = True
            sys.stdout.write(f"File contains: {self.exptRunDict.keys()}, Setup Ok: {self.setupOk}\n")
            sys.stdout.flush()

    def runExpt(self, clicked):
        if self.setupOk:
            # run the experiment
            self.exptRun.acquireEvents = self.exptRunDict['setup']['events']
            self.exptRun.imageProcessParameters = {
                'imageHeight': self.exptRunDict['analysis']["imageHeight"],
                'imageWidth': self.exptRunDict['analysis']["imageWidth"],
                'cellModelPath': self.exptRunDict['analysis']["cellSegNetModelPath"],
                'channelModelPath': self.exptRunDict['analysis']["channelSegNetModelPath"],
                'saveDir': self.exptRunDict['analysis']["saveDir"]
            }

            self.exptRun.dbParameters = self.exptRunDict['database']

            self.exptRun.setImageTransforms()

            runProcesses(self.exptRun)
            self.exptRunStarted = True


            sys.stdout.write(f"Expt setup ok ... Running now:)\n")
            sys.stdout.flush()


    def stopExpt(self, clicked):

        if self.exptRunStarted and self.setupOk:
            self.exptRun.stop()

        sys.stdout.write(f"Run stopped ... :(\n")
        sys.stdout.flush()

    

