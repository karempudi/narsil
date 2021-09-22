# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'setupWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.1.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *  # type: ignore
from PySide6.QtGui import *  # type: ignore
from PySide6.QtWidgets import *  # type: ignore


class Ui_SetupWindow(object):
    def setupUi(self, SetupWindow):
        if not SetupWindow.objectName():
            SetupWindow.setObjectName(u"SetupWindow")
        SetupWindow.resize(594, 593)
        self.centralwidget = QWidget(SetupWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.exptSetupBox = QGroupBox(self.centralwidget)
        self.exptSetupBox.setObjectName(u"exptSetupBox")
        self.exptSetupBox.setGeometry(QRect(30, 10, 541, 231))
        self.horizontalLayoutWidget_3 = QWidget(self.exptSetupBox)
        self.horizontalLayoutWidget_3.setObjectName(u"horizontalLayoutWidget_3")
        self.horizontalLayoutWidget_3.setGeometry(QRect(10, 20, 431, 41))
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
        self.horizontalLayoutWidget_4.setGeometry(QRect(10, 60, 431, 41))
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
        self.horizontalLayoutWidget_5.setGeometry(QRect(10, 100, 431, 41))
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
        self.horizontalLayoutWidget_6.setGeometry(QRect(10, 150, 431, 41))
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
        self.validateExptSetupButton.setGeometry(QRect(260, 200, 171, 23))
        self.analysisSetupBox = QGroupBox(self.centralwidget)
        self.analysisSetupBox.setObjectName(u"analysisSetupBox")
        self.analysisSetupBox.setGeometry(QRect(30, 250, 541, 211))
        self.horizontalLayoutWidget = QWidget(self.analysisSetupBox)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(10, 40, 431, 31))
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
        self.horizontalLayoutWidget_7.setGeometry(QRect(10, 80, 431, 31))
        self.channelSegLayout = QHBoxLayout(self.horizontalLayoutWidget_7)
        self.channelSegLayout.setObjectName(u"channelSegLayout")
        self.channelSegLayout.setContentsMargins(0, 0, 0, 0)
        self.segChannels = QCheckBox(self.horizontalLayoutWidget_7)
        self.segChannels.setObjectName(u"segChannels")

        self.channelSegLayout.addWidget(self.segChannels)

        self.horizontalLayoutWidget_8 = QWidget(self.analysisSetupBox)
        self.horizontalLayoutWidget_8.setObjectName(u"horizontalLayoutWidget_8")
        self.horizontalLayoutWidget_8.setGeometry(QRect(10, 120, 431, 31))
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
        self.validateAnalysisSetupButton.setGeometry(QRect(280, 170, 131, 23))
        self.horizontalLayoutWidget_2 = QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setObjectName(u"horizontalLayoutWidget_2")
        self.horizontalLayoutWidget_2.setGeometry(QRect(270, 470, 221, 51))
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
        self.menubar.setGeometry(QRect(0, 0, 594, 21))
        SetupWindow.setMenuBar(self.menubar)

        self.retranslateUi(SetupWindow)

        QMetaObject.connectSlotsByName(SetupWindow)
    # setupUi

    def retranslateUi(self, SetupWindow):
        SetupWindow.setWindowTitle(QCoreApplication.translate("SetupWindow", u"SetupWindow", None))
        self.exptSetupBox.setTitle(QCoreApplication.translate("SetupWindow", u"Experiment Setup", None))
        self.exptNoLabel.setText(QCoreApplication.translate("SetupWindow", u"Experiment No:", None))
        self.exptNoText.setText(QCoreApplication.translate("SetupWindow", u"EXP-21-BP000", None))
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
        self.analysisSetupBox.setTitle(QCoreApplication.translate("SetupWindow", u"Analysis Setup", None))
        self.segmentationLabel.setText(QCoreApplication.translate("SetupWindow", u"Cell Segmentation: ", None))
        self.selectNet.setItemText(0, QCoreApplication.translate("SetupWindow", u"Normal U-net", None))
        self.selectNet.setItemText(1, QCoreApplication.translate("SetupWindow", u"Small U-net", None))

        self.segChannels.setText(QCoreApplication.translate("SetupWindow", u"Channel Segmentation", None))
        self.calcDeadAlive.setText(QCoreApplication.translate("SetupWindow", u"DeadAlive", None))
        self.calcGrowthRates.setText(QCoreApplication.translate("SetupWindow", u"Growth Rates", None))
        self.validateAnalysisSetupButton.setText(QCoreApplication.translate("SetupWindow", u"Validate Analysis Setup", None))
        self.resetExptSetupButton.setText(QCoreApplication.translate("SetupWindow", u"Reset", None))
        self.closeExptSetupButton.setText(QCoreApplication.translate("SetupWindow", u"Close", None))
    # retranslateUi

