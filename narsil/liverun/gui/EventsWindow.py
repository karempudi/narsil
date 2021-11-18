


from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PySide6.QtCore import QFile, QIODevice, QTimer, Signal, Qt, QThread
from PySide6.QtUiTools import QUiLoader


from narsil.liverun.ui.ui_EventsWindow import Ui_EventsWindow
from narsil.liverun.utils import getPositionList
from collections import OrderedDict

from narsil.liverun.utils import getPositionsMicroManager, getMicroManagerPresets, imgFilenameFromNumber

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

        # micromanager version
        self.mmVersion = None

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
            self.positionsData = getPositionList(filename=self.positionsFileName, version=self.mmVersion)
        
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

