from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PySide6.QtCore import QFile, QIODevice, QTimer, Signal, Qt, QThread 
from PySide6.QtGui import QIntValidator
from PySide6.QtUiTools import QUiLoader

from narsil.liverun.ui.ui_RunWindow import Ui_runWindow

from pathlib import Path
from skimage import io

import numpy as np
import sys
import glob


class RunWindow(QMainWindow):

    def __init__(self):
        super(RunWindow, self).__init__()
        self.ui = Ui_runWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Experiment Run window")


        self.exptRunDict = None

        self.setupButtonHandlers()
    
    def setupButtonHandlers(self):
        
        # load the data from the experiment file
        self.ui.loadButton.clicked.connect(self.loadExptRun)

        # Run the experiment
        self.ui.runButton.clicked.connect(self.runExpt)

        self.ui.stopButton.clicked.connect(self.stopExpt)

    def loadExptRun(self, clicked):
        try:
            pass
        except Exception as e:
            sys.stdout.write(f"Error in loading the experimental setup file -- {e}\n")
            sys.stdout.flush()
        finally:
            pass

    def runExpt(self, clicked):
        sys.stdout.write(f"Run started ... :)\n")
        sys.stdout.flush()

    def stopExpt(self, clicked):
        sys.stdout.write(f"Run stopped ... :(\n")
        sys.stdout.flush()

    

