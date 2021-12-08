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

        self.setupButtonHandlers()
    
    def setupButtonHandlers(self):
        
        # load the data from the experiment file
        self.ui.loadButton.clicked.connect(self.loadExptRun)

        # Run the experiment
        self.ui.runButton.clicked.connect(self.runExpt)

        self.ui.stopButton.clicked.connect(self.stopExpt)

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

    

