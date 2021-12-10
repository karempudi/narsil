# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'viewerWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.2.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMenuBar,
    QPushButton, QRadioButton, QSizePolicy, QSlider,
    QStatusBar, QWidget)

from pyqtgraph import (ImageView, PlotWidget)

class Ui_ViewerWindow(object):
    def setupUi(self, ViewerWindow):
        if not ViewerWindow.objectName():
            ViewerWindow.setObjectName(u"ViewerWindow")
        ViewerWindow.resize(1577, 736)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(ViewerWindow.sizePolicy().hasHeightForWidth())
        ViewerWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QWidget(ViewerWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.removeButton = QPushButton(self.centralwidget)
        self.removeButton.setObjectName(u"removeButton")
        self.removeButton.setGeometry(QRect(1090, 450, 89, 25))
        self.showButton = QPushButton(self.centralwidget)
        self.showButton.setObjectName(u"showButton")
        self.showButton.setGeometry(QRect(990, 450, 89, 25))
        self.undoButton = QPushButton(self.centralwidget)
        self.undoButton.setObjectName(u"undoButton")
        self.undoButton.setGeometry(QRect(990, 490, 89, 25))
        self.resetButton = QPushButton(self.centralwidget)
        self.resetButton.setObjectName(u"resetButton")
        self.resetButton.setGeometry(QRect(1090, 490, 89, 25))
        self.imagePlot = ImageView(self.centralwidget)
        self.imagePlot.setObjectName(u"imagePlot")
        self.imagePlot.setGeometry(QRect(20, 110, 541, 561))
        self.filterParametersBox = QGroupBox(self.centralwidget)
        self.filterParametersBox.setObjectName(u"filterParametersBox")
        self.filterParametersBox.setGeometry(QRect(610, 80, 341, 261))
        self.formLayoutWidget = QWidget(self.filterParametersBox)
        self.formLayoutWidget.setObjectName(u"formLayoutWidget")
        self.formLayoutWidget.setGeometry(QRect(30, 30, 261, 201))
        self.formLayout = QFormLayout(self.formLayoutWidget)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.windowLengthLabel = QLabel(self.formLayoutWidget)
        self.windowLengthLabel.setObjectName(u"windowLengthLabel")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.windowLengthLabel)

        self.windowLengthLine = QLineEdit(self.formLayoutWidget)
        self.windowLengthLine.setObjectName(u"windowLengthLine")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.windowLengthLine)

        self.areaLabel = QLabel(self.formLayoutWidget)
        self.areaLabel.setObjectName(u"areaLabel")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.areaLabel)

        self.lengthLabel = QLabel(self.formLayoutWidget)
        self.lengthLabel.setObjectName(u"lengthLabel")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.lengthLabel)

        self.areaSlider = QSlider(self.formLayoutWidget)
        self.areaSlider.setObjectName(u"areaSlider")
        self.areaSlider.setOrientation(Qt.Horizontal)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.areaSlider)

        self.lengthSlider = QSlider(self.formLayoutWidget)
        self.lengthSlider.setObjectName(u"lengthSlider")
        self.lengthSlider.setOrientation(Qt.Horizontal)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.lengthSlider)

        self.cellObjectsLabel = QLabel(self.formLayoutWidget)
        self.cellObjectsLabel.setObjectName(u"cellObjectsLabel")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.cellObjectsLabel)

        self.cellObjectsSlider = QSlider(self.formLayoutWidget)
        self.cellObjectsSlider.setObjectName(u"cellObjectsSlider")
        self.cellObjectsSlider.setOrientation(Qt.Horizontal)

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.cellObjectsSlider)

        self.nextAutoButton = QPushButton(self.centralwidget)
        self.nextAutoButton.setObjectName(u"nextAutoButton")
        self.nextAutoButton.setGeometry(QRect(1090, 530, 89, 25))
        self.sendTweezePositionsButton = QPushButton(self.centralwidget)
        self.sendTweezePositionsButton.setObjectName(u"sendTweezePositionsButton")
        self.sendTweezePositionsButton.setGeometry(QRect(980, 630, 191, 25))
        self.propertiesView = PlotWidget(self.centralwidget)
        self.propertiesView.setObjectName(u"propertiesView")
        self.propertiesView.setGeometry(QRect(610, 410, 321, 241))
        self.horizontalLayoutWidget = QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(30, 10, 401, 41))
        self.horizontalLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.positionLabel = QLabel(self.horizontalLayoutWidget)
        self.positionLabel.setObjectName(u"positionLabel")

        self.horizontalLayout.addWidget(self.positionLabel)

        self.positionNoLine = QLineEdit(self.horizontalLayoutWidget)
        self.positionNoLine.setObjectName(u"positionNoLine")

        self.horizontalLayout.addWidget(self.positionNoLine)

        self.channelNoLabel = QLabel(self.horizontalLayoutWidget)
        self.channelNoLabel.setObjectName(u"channelNoLabel")

        self.horizontalLayout.addWidget(self.channelNoLabel)

        self.channelNoLine = QLineEdit(self.horizontalLayoutWidget)
        self.channelNoLine.setObjectName(u"channelNoLine")

        self.horizontalLayout.addWidget(self.channelNoLine)

        self.fetchButton = QPushButton(self.horizontalLayoutWidget)
        self.fetchButton.setObjectName(u"fetchButton")

        self.horizontalLayout.addWidget(self.fetchButton)

        self.horizontalLayoutWidget_2 = QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setObjectName(u"horizontalLayoutWidget_2")
        self.horizontalLayoutWidget_2.setGeometry(QRect(30, 60, 491, 41))
        self.horizontalLayout_2 = QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.phaseImage = QRadioButton(self.horizontalLayoutWidget_2)
        self.phaseImage.setObjectName(u"phaseImage")
        self.phaseImage.setChecked(True)

        self.horizontalLayout_2.addWidget(self.phaseImage)

        self.cellSegImage = QRadioButton(self.horizontalLayoutWidget_2)
        self.cellSegImage.setObjectName(u"cellSegImage")

        self.horizontalLayout_2.addWidget(self.cellSegImage)

        self.horizontalLayoutWidget_3 = QWidget(self.centralwidget)
        self.horizontalLayoutWidget_3.setObjectName(u"horizontalLayoutWidget_3")
        self.horizontalLayoutWidget_3.setGeometry(QRect(570, 340, 391, 61))
        self.horizontalLayout_3 = QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.pushButton = QPushButton(self.horizontalLayoutWidget_3)
        self.pushButton.setObjectName(u"pushButton")

        self.horizontalLayout_3.addWidget(self.pushButton)

        self.findLocationsButton = QPushButton(self.horizontalLayoutWidget_3)
        self.findLocationsButton.setObjectName(u"findLocationsButton")

        self.horizontalLayout_3.addWidget(self.findLocationsButton)

        self.isExptRunning = QCheckBox(self.centralwidget)
        self.isExptRunning.setObjectName(u"isExptRunning")
        self.isExptRunning.setGeometry(QRect(460, 20, 131, 23))
        self.activePositionsList = QListWidget(self.centralwidget)
        self.activePositionsList.setObjectName(u"activePositionsList")
        self.activePositionsList.setGeometry(QRect(970, 10, 256, 421))
        self.activePositionsList.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tweezePositionsList = QListWidget(self.centralwidget)
        self.tweezePositionsList.setObjectName(u"tweezePositionsList")
        self.tweezePositionsList.setGeometry(QRect(1280, 10, 256, 421))
        self.tweezePositionsList.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.toTweezeListButton = QPushButton(self.centralwidget)
        self.toTweezeListButton.setObjectName(u"toTweezeListButton")
        self.toTweezeListButton.setGeometry(QRect(1230, 210, 41, 21))
        self.toActiveListButton = QPushButton(self.centralwidget)
        self.toActiveListButton.setObjectName(u"toActiveListButton")
        self.toActiveListButton.setGeometry(QRect(1230, 270, 41, 21))
        self.viewActiveListCheck = QCheckBox(self.centralwidget)
        self.viewActiveListCheck.setObjectName(u"viewActiveListCheck")
        self.viewActiveListCheck.setGeometry(QRect(1000, 580, 92, 23))
        self.viewActiveListCheck.setChecked(True)
        self.horizontalLayoutWidget_4 = QWidget(self.centralwidget)
        self.horizontalLayoutWidget_4.setObjectName(u"horizontalLayoutWidget_4")
        self.horizontalLayoutWidget_4.setGeometry(QRect(620, 10, 230, 41))
        self.horizontalLayout_4 = QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.getOnly20Radio = QRadioButton(self.horizontalLayoutWidget_4)
        self.getOnly20Radio.setObjectName(u"getOnly20Radio")
        self.getOnly20Radio.setChecked(True)

        self.horizontalLayout_4.addWidget(self.getOnly20Radio)

        self.getAllImagesRadio = QRadioButton(self.horizontalLayoutWidget_4)
        self.getAllImagesRadio.setObjectName(u"getAllImagesRadio")

        self.horizontalLayout_4.addWidget(self.getAllImagesRadio)

        self.plotPropertiesCheck = QCheckBox(self.centralwidget)
        self.plotPropertiesCheck.setObjectName(u"plotPropertiesCheck")
        self.plotPropertiesCheck.setGeometry(QRect(1090, 580, 121, 23))
        ViewerWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(ViewerWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1577, 22))
        ViewerWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(ViewerWindow)
        self.statusbar.setObjectName(u"statusbar")
        ViewerWindow.setStatusBar(self.statusbar)

        self.retranslateUi(ViewerWindow)

        QMetaObject.connectSlotsByName(ViewerWindow)
    # setupUi

    def retranslateUi(self, ViewerWindow):
        ViewerWindow.setWindowTitle(QCoreApplication.translate("ViewerWindow", u"MainWindow", None))
        self.removeButton.setText(QCoreApplication.translate("ViewerWindow", u"Remove", None))
        self.showButton.setText(QCoreApplication.translate("ViewerWindow", u"Show", None))
        self.undoButton.setText(QCoreApplication.translate("ViewerWindow", u"Undo", None))
        self.resetButton.setText(QCoreApplication.translate("ViewerWindow", u"Reset", None))
        self.filterParametersBox.setTitle(QCoreApplication.translate("ViewerWindow", u"Filter Parameters", None))
        self.windowLengthLabel.setText(QCoreApplication.translate("ViewerWindow", u"Rolling window Length", None))
        self.areaLabel.setText(QCoreApplication.translate("ViewerWindow", u"Area Threshold", None))
        self.lengthLabel.setText(QCoreApplication.translate("ViewerWindow", u"Length Threshold", None))
        self.cellObjectsLabel.setText(QCoreApplication.translate("ViewerWindow", u"No of Cell like objects", None))
        self.nextAutoButton.setText(QCoreApplication.translate("ViewerWindow", u"Next Auto", None))
        self.sendTweezePositionsButton.setText(QCoreApplication.translate("ViewerWindow", u"Send Tweeze Positions", None))
        self.positionLabel.setText(QCoreApplication.translate("ViewerWindow", u"Position", None))
        self.channelNoLabel.setText(QCoreApplication.translate("ViewerWindow", u"Channel No", None))
        self.fetchButton.setText(QCoreApplication.translate("ViewerWindow", u"Fetch", None))
        self.phaseImage.setText(QCoreApplication.translate("ViewerWindow", u"Phase", None))
        self.cellSegImage.setText(QCoreApplication.translate("ViewerWindow", u"Cell Seg", None))
        self.pushButton.setText(QCoreApplication.translate("ViewerWindow", u"Update Filter Parameters", None))
        self.findLocationsButton.setText(QCoreApplication.translate("ViewerWindow", u"Find All Tweezable Channels", None))
        self.isExptRunning.setText(QCoreApplication.translate("ViewerWindow", u"Is Expt running?", None))
        self.toTweezeListButton.setText(QCoreApplication.translate("ViewerWindow", u">>", None))
        self.toActiveListButton.setText(QCoreApplication.translate("ViewerWindow", u"<<", None))
        self.viewActiveListCheck.setText(QCoreApplication.translate("ViewerWindow", u"Active? ", None))
        self.getOnly20Radio.setText(QCoreApplication.translate("ViewerWindow", u"Last 20 images", None))
        self.getAllImagesRadio.setText(QCoreApplication.translate("ViewerWindow", u"All Images", None))
        self.plotPropertiesCheck.setText(QCoreApplication.translate("ViewerWindow", u"Plot Properties", None))
    # retranslateUi

