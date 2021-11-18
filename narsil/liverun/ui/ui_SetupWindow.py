# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'setupWindow.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMenuBar, QPushButton, QRadioButton,
    QSizePolicy, QStatusBar, QWidget)

class Ui_SetupWindow(object):
    def setupUi(self, SetupWindow):
        if not SetupWindow.objectName():
            SetupWindow.setObjectName(u"SetupWindow")
        SetupWindow.resize(548, 739)
        self.centralwidget = QWidget(SetupWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.exptSetupBox = QGroupBox(self.centralwidget)
        self.exptSetupBox.setObjectName(u"exptSetupBox")
        self.exptSetupBox.setGeometry(QRect(20, 10, 501, 271))
        self.horizontalLayoutWidget_3 = QWidget(self.exptSetupBox)
        self.horizontalLayoutWidget_3.setObjectName(u"horizontalLayoutWidget_3")
        self.horizontalLayoutWidget_3.setGeometry(QRect(10, 30, 431, 41))
        self.exptNoLayout = QHBoxLayout(self.horizontalLayoutWidget_3)
        self.exptNoLayout.setObjectName(u"exptNoLayout")
        self.exptNoLayout.setContentsMargins(0, 0, 0, 0)
        self.exptNoLabel = QLabel(self.horizontalLayoutWidget_3)
        self.exptNoLabel.setObjectName(u"exptNoLabel")

        self.exptNoLayout.addWidget(self.exptNoLabel)

        self.exptNoText = QLineEdit(self.horizontalLayoutWidget_3)
        self.exptNoText.setObjectName(u"exptNoText")

        self.exptNoLayout.addWidget(self.exptNoText)

        self.exptNoSetButton = QPushButton(self.horizontalLayoutWidget_3)
        self.exptNoSetButton.setObjectName(u"exptNoSetButton")

        self.exptNoLayout.addWidget(self.exptNoSetButton)

        self.exptNoClearButton = QPushButton(self.horizontalLayoutWidget_3)
        self.exptNoClearButton.setObjectName(u"exptNoClearButton")

        self.exptNoLayout.addWidget(self.exptNoClearButton)

        self.horizontalLayoutWidget_4 = QWidget(self.exptSetupBox)
        self.horizontalLayoutWidget_4.setObjectName(u"horizontalLayoutWidget_4")
        self.horizontalLayoutWidget_4.setGeometry(QRect(10, 70, 431, 41))
        self.positionsInputLayout = QHBoxLayout(self.horizontalLayoutWidget_4)
        self.positionsInputLayout.setObjectName(u"positionsInputLayout")
        self.positionsInputLayout.setContentsMargins(0, 0, 0, 0)
        self.positionsInputLabel = QLabel(self.horizontalLayoutWidget_4)
        self.positionsInputLabel.setObjectName(u"positionsInputLabel")

        self.positionsInputLayout.addWidget(self.positionsInputLabel)

        self.fromFile = QRadioButton(self.horizontalLayoutWidget_4)
        self.fromFile.setObjectName(u"fromFile")
        self.fromFile.setCheckable(True)
        self.fromFile.setChecked(False)

        self.positionsInputLayout.addWidget(self.fromFile)

        self.fromMicroManager = QRadioButton(self.horizontalLayoutWidget_4)
        self.fromMicroManager.setObjectName(u"fromMicroManager")
        self.fromMicroManager.setEnabled(True)
        self.fromMicroManager.setCheckable(True)
        self.fromMicroManager.setChecked(False)

        self.positionsInputLayout.addWidget(self.fromMicroManager)

        self.horizontalLayoutWidget_5 = QWidget(self.exptSetupBox)
        self.horizontalLayoutWidget_5.setObjectName(u"horizontalLayoutWidget_5")
        self.horizontalLayoutWidget_5.setGeometry(QRect(10, 110, 431, 41))
        self.fileSelectionLayout = QHBoxLayout(self.horizontalLayoutWidget_5)
        self.fileSelectionLayout.setObjectName(u"fileSelectionLayout")
        self.fileSelectionLayout.setContentsMargins(0, 0, 0, 0)
        self.fileSelectionLabel = QLabel(self.horizontalLayoutWidget_5)
        self.fileSelectionLabel.setObjectName(u"fileSelectionLabel")

        self.fileSelectionLayout.addWidget(self.fileSelectionLabel)

        self.fileSelectionButton = QPushButton(self.horizontalLayoutWidget_5)
        self.fileSelectionButton.setObjectName(u"fileSelectionButton")

        self.fileSelectionLayout.addWidget(self.fileSelectionButton)

        self.horizontalLayoutWidget_6 = QWidget(self.exptSetupBox)
        self.horizontalLayoutWidget_6.setObjectName(u"horizontalLayoutWidget_6")
        self.horizontalLayoutWidget_6.setGeometry(QRect(10, 190, 431, 41))
        self.eventsCreationLayout = QHBoxLayout(self.horizontalLayoutWidget_6)
        self.eventsCreationLayout.setObjectName(u"eventsCreationLayout")
        self.eventsCreationLayout.setContentsMargins(0, 0, 0, 0)
        self.eventsCreationLabel = QLabel(self.horizontalLayoutWidget_6)
        self.eventsCreationLabel.setObjectName(u"eventsCreationLabel")

        self.eventsCreationLayout.addWidget(self.eventsCreationLabel)

        self.eventsCreationButton = QPushButton(self.horizontalLayoutWidget_6)
        self.eventsCreationButton.setObjectName(u"eventsCreationButton")

        self.eventsCreationLayout.addWidget(self.eventsCreationButton)

        self.validateExptSetupButton = QPushButton(self.exptSetupBox)
        self.validateExptSetupButton.setObjectName(u"validateExptSetupButton")
        self.validateExptSetupButton.setGeometry(QRect(240, 240, 191, 23))
        self.horizontalLayoutWidget_9 = QWidget(self.exptSetupBox)
        self.horizontalLayoutWidget_9.setObjectName(u"horizontalLayoutWidget_9")
        self.horizontalLayoutWidget_9.setGeometry(QRect(10, 150, 431, 41))
        self.mmVersionLayout = QHBoxLayout(self.horizontalLayoutWidget_9)
        self.mmVersionLayout.setObjectName(u"mmVersionLayout")
        self.mmVersionLayout.setContentsMargins(0, 0, 0, 0)
        self.microManagerVersionLabel = QLabel(self.horizontalLayoutWidget_9)
        self.microManagerVersionLabel.setObjectName(u"microManagerVersionLabel")

        self.mmVersionLayout.addWidget(self.microManagerVersionLabel)

        self.version1Button = QRadioButton(self.horizontalLayoutWidget_9)
        self.version1Button.setObjectName(u"version1Button")
        self.version1Button.setCheckable(True)
        self.version1Button.setChecked(False)

        self.mmVersionLayout.addWidget(self.version1Button)

        self.version2Button = QRadioButton(self.horizontalLayoutWidget_9)
        self.version2Button.setObjectName(u"version2Button")
        self.version2Button.setEnabled(True)
        self.version2Button.setCheckable(True)
        self.version2Button.setChecked(False)

        self.mmVersionLayout.addWidget(self.version2Button)

        self.analysisSetupBox = QGroupBox(self.centralwidget)
        self.analysisSetupBox.setObjectName(u"analysisSetupBox")
        self.analysisSetupBox.setGeometry(QRect(20, 290, 511, 331))
        self.horizontalLayoutWidget = QWidget(self.analysisSetupBox)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(10, 30, 431, 31))
        self.segmentationLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.segmentationLayout.setObjectName(u"segmentationLayout")
        self.segmentationLayout.setContentsMargins(0, 0, 0, 0)
        self.segmentationLabel = QLabel(self.horizontalLayoutWidget)
        self.segmentationLabel.setObjectName(u"segmentationLabel")

        self.segmentationLayout.addWidget(self.segmentationLabel)

        self.selectNet = QComboBox(self.horizontalLayoutWidget)
        self.selectNet.addItem("")
        self.selectNet.addItem("")
        self.selectNet.setObjectName(u"selectNet")

        self.segmentationLayout.addWidget(self.selectNet)

        self.horizontalLayoutWidget_7 = QWidget(self.analysisSetupBox)
        self.horizontalLayoutWidget_7.setObjectName(u"horizontalLayoutWidget_7")
        self.horizontalLayoutWidget_7.setGeometry(QRect(10, 230, 431, 31))
        self.channelSegLayout = QHBoxLayout(self.horizontalLayoutWidget_7)
        self.channelSegLayout.setObjectName(u"channelSegLayout")
        self.channelSegLayout.setContentsMargins(0, 0, 0, 0)
        self.segChannels = QCheckBox(self.horizontalLayoutWidget_7)
        self.segChannels.setObjectName(u"segChannels")

        self.channelSegLayout.addWidget(self.segChannels)

        self.horizontalLayoutWidget_8 = QWidget(self.analysisSetupBox)
        self.horizontalLayoutWidget_8.setObjectName(u"horizontalLayoutWidget_8")
        self.horizontalLayoutWidget_8.setGeometry(QRect(10, 260, 431, 31))
        self.cellAnalysisLayout = QHBoxLayout(self.horizontalLayoutWidget_8)
        self.cellAnalysisLayout.setObjectName(u"cellAnalysisLayout")
        self.cellAnalysisLayout.setContentsMargins(0, 0, 0, 0)
        self.calcDeadAlive = QCheckBox(self.horizontalLayoutWidget_8)
        self.calcDeadAlive.setObjectName(u"calcDeadAlive")

        self.cellAnalysisLayout.addWidget(self.calcDeadAlive)

        self.calcGrowthRates = QCheckBox(self.horizontalLayoutWidget_8)
        self.calcGrowthRates.setObjectName(u"calcGrowthRates")

        self.cellAnalysisLayout.addWidget(self.calcGrowthRates)

        self.validateAnalysisSetupButton = QPushButton(self.analysisSetupBox)
        self.validateAnalysisSetupButton.setObjectName(u"validateAnalysisSetupButton")
        self.validateAnalysisSetupButton.setGeometry(QRect(270, 300, 161, 23))
        self.formLayoutWidget = QWidget(self.analysisSetupBox)
        self.formLayoutWidget.setObjectName(u"formLayoutWidget")
        self.formLayoutWidget.setGeometry(QRect(10, 60, 479, 161))
        self.imageSegLayout = QFormLayout(self.formLayoutWidget)
        self.imageSegLayout.setObjectName(u"imageSegLayout")
        self.imageSegLayout.setContentsMargins(0, 0, 0, 0)
        self.cellSegLabel = QLabel(self.formLayoutWidget)
        self.cellSegLabel.setObjectName(u"cellSegLabel")

        self.imageSegLayout.setWidget(0, QFormLayout.LabelRole, self.cellSegLabel)

        self.cellSegNetFilePathButton = QPushButton(self.formLayoutWidget)
        self.cellSegNetFilePathButton.setObjectName(u"cellSegNetFilePathButton")

        self.imageSegLayout.setWidget(0, QFormLayout.FieldRole, self.cellSegNetFilePathButton)

        self.channelSegLabel = QLabel(self.formLayoutWidget)
        self.channelSegLabel.setObjectName(u"channelSegLabel")

        self.imageSegLayout.setWidget(1, QFormLayout.LabelRole, self.channelSegLabel)

        self.channelSegNetFilePathButton = QPushButton(self.formLayoutWidget)
        self.channelSegNetFilePathButton.setObjectName(u"channelSegNetFilePathButton")

        self.imageSegLayout.setWidget(1, QFormLayout.FieldRole, self.channelSegNetFilePathButton)

        self.imageWidthLabel = QLabel(self.formLayoutWidget)
        self.imageWidthLabel.setObjectName(u"imageWidthLabel")

        self.imageSegLayout.setWidget(4, QFormLayout.LabelRole, self.imageWidthLabel)

        self.imageWidth = QLineEdit(self.formLayoutWidget)
        self.imageWidth.setObjectName(u"imageWidth")

        self.imageSegLayout.setWidget(4, QFormLayout.FieldRole, self.imageWidth)

        self.imageHeight = QLineEdit(self.formLayoutWidget)
        self.imageHeight.setObjectName(u"imageHeight")

        self.imageSegLayout.setWidget(2, QFormLayout.FieldRole, self.imageHeight)

        self.imageHeightLabel = QLabel(self.formLayoutWidget)
        self.imageHeightLabel.setObjectName(u"imageHeightLabel")

        self.imageSegLayout.setWidget(2, QFormLayout.LabelRole, self.imageHeightLabel)

        self.savePathLabel = QLabel(self.formLayoutWidget)
        self.savePathLabel.setObjectName(u"savePathLabel")

        self.imageSegLayout.setWidget(5, QFormLayout.LabelRole, self.savePathLabel)

        self.saveDirButton = QPushButton(self.formLayoutWidget)
        self.saveDirButton.setObjectName(u"saveDirButton")

        self.imageSegLayout.setWidget(5, QFormLayout.FieldRole, self.saveDirButton)

        self.horizontalLayoutWidget_2 = QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setObjectName(u"horizontalLayoutWidget_2")
        self.horizontalLayoutWidget_2.setGeometry(QRect(260, 630, 221, 51))
        self.exptSaveCloseButtons = QHBoxLayout(self.horizontalLayoutWidget_2)
        self.exptSaveCloseButtons.setObjectName(u"exptSaveCloseButtons")
        self.exptSaveCloseButtons.setContentsMargins(0, 0, 0, 0)
        self.resetExptSetupButton = QPushButton(self.horizontalLayoutWidget_2)
        self.resetExptSetupButton.setObjectName(u"resetExptSetupButton")

        self.exptSaveCloseButtons.addWidget(self.resetExptSetupButton)

        self.closeExptSetupButton = QPushButton(self.horizontalLayoutWidget_2)
        self.closeExptSetupButton.setObjectName(u"closeExptSetupButton")

        self.exptSaveCloseButtons.addWidget(self.closeExptSetupButton)

        SetupWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(SetupWindow)
        self.statusbar.setObjectName(u"statusbar")
        SetupWindow.setStatusBar(self.statusbar)
        self.menubar = QMenuBar(SetupWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 548, 22))
        SetupWindow.setMenuBar(self.menubar)

        self.retranslateUi(SetupWindow)

        QMetaObject.connectSlotsByName(SetupWindow)
    # setupUi

    def retranslateUi(self, SetupWindow):
        SetupWindow.setWindowTitle(QCoreApplication.translate("SetupWindow", u"SetupWindow", None))
        self.exptSetupBox.setTitle(QCoreApplication.translate("SetupWindow", u"Experiment Setup", None))
        self.exptNoLabel.setText(QCoreApplication.translate("SetupWindow", u"Experiment No:", None))
        self.exptNoText.setText(QCoreApplication.translate("SetupWindow", u"EXP21BP000", None))
        self.exptNoSetButton.setText(QCoreApplication.translate("SetupWindow", u"Set", None))
        self.exptNoClearButton.setText(QCoreApplication.translate("SetupWindow", u"Clear", None))
        self.positionsInputLabel.setText(QCoreApplication.translate("SetupWindow", u"Get Positions from:", None))
        self.fromFile.setText(QCoreApplication.translate("SetupWindow", u"File", None))
        self.fromMicroManager.setText(QCoreApplication.translate("SetupWindow", u"Micromanager", None))
        self.fileSelectionLabel.setText(QCoreApplication.translate("SetupWindow", u"If File Select File:", None))
        self.fileSelectionButton.setText(QCoreApplication.translate("SetupWindow", u"Select File", None))
        self.eventsCreationLabel.setText(QCoreApplication.translate("SetupWindow", u"Events Creation:", None))
        self.eventsCreationButton.setText(QCoreApplication.translate("SetupWindow", u"Create Events", None))
        self.validateExptSetupButton.setText(QCoreApplication.translate("SetupWindow", u"Validate Experiment Setup", None))
        self.microManagerVersionLabel.setText(QCoreApplication.translate("SetupWindow", u"Micromanager Version", None))
        self.version1Button.setText(QCoreApplication.translate("SetupWindow", u"1.4", None))
        self.version2Button.setText(QCoreApplication.translate("SetupWindow", u"2.0", None))
        self.analysisSetupBox.setTitle(QCoreApplication.translate("SetupWindow", u"Analysis Setup", None))
        self.segmentationLabel.setText(QCoreApplication.translate("SetupWindow", u"Cell Segmentation: ", None))
        self.selectNet.setItemText(0, QCoreApplication.translate("SetupWindow", u"normal", None))
        self.selectNet.setItemText(1, QCoreApplication.translate("SetupWindow", u"small", None))

        self.segChannels.setText(QCoreApplication.translate("SetupWindow", u"Channel Segmentation", None))
        self.calcDeadAlive.setText(QCoreApplication.translate("SetupWindow", u"DeadAlive", None))
        self.calcGrowthRates.setText(QCoreApplication.translate("SetupWindow", u"Growth Rates", None))
        self.validateAnalysisSetupButton.setText(QCoreApplication.translate("SetupWindow", u"Validate Analysis Setup", None))
        self.cellSegLabel.setText(QCoreApplication.translate("SetupWindow", u"Cell seg Net Path:                         ", None))
        self.cellSegNetFilePathButton.setText(QCoreApplication.translate("SetupWindow", u"Select Cell SegNet model file", None))
        self.channelSegLabel.setText(QCoreApplication.translate("SetupWindow", u"Channel seg Net Path:                         ", None))
        self.channelSegNetFilePathButton.setText(QCoreApplication.translate("SetupWindow", u"Select Channel SegNet model file", None))
        self.imageWidthLabel.setText(QCoreApplication.translate("SetupWindow", u"Image Width:", None))
        self.imageHeightLabel.setText(QCoreApplication.translate("SetupWindow", u"Image Height:", None))
        self.savePathLabel.setText(QCoreApplication.translate("SetupWindow", u"Save Path:", None))
        self.saveDirButton.setText(QCoreApplication.translate("SetupWindow", u"Select directory for saving", None))
        self.resetExptSetupButton.setText(QCoreApplication.translate("SetupWindow", u"Reset", None))
        self.closeExptSetupButton.setText(QCoreApplication.translate("SetupWindow", u"Close", None))
    # retranslateUi

