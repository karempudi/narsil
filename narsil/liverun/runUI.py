# File for starting the running UI for the experiment
# where you can load the settings that you created using exptUI.py
# and start all the processes that do the calculations and stop them

from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PySide6.QtCore import QFile, QIODevice, QTimer, Signal, Qt, QThread

import sys
from narsil.liverun.gui.runWindow import RunWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RunWindow()
    window.show()
    sys.exit(app.exec())