# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'eventsWindow.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMenuBar, QPushButton,
    QSizePolicy, QSpinBox, QStatusBar, QVBoxLayout,
    QWidget)

class Ui_EventsWindow(object):
    def setupUi(self, EventsWindow):
        if not EventsWindow.objectName():
            EventsWindow.setObjectName(u"EventsWindow")
        EventsWindow.resize(685, 733)
        self.centralwidget = QWidget(EventsWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.FastPositionsLabel = QLabel(self.centralwidget)
        self.FastPositionsLabel.setObjectName(u"FastPositionsLabel")
        self.FastPositionsLabel.setGeometry(QRect(30, 20, 171, 19))
        self.horizontalLayoutWidget = QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(30, 270, 356, 80))
        self.positionsButtonsLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.positionsButtonsLayout.setObjectName(u"positionsButtonsLayout")
        self.positionsButtonsLayout.setContentsMargins(0, 0, 0, 0)
        self.getPositionsButton = QPushButton(self.horizontalLayoutWidget)
        self.getPositionsButton.setObjectName(u"getPositionsButton")

        self.positionsButtonsLayout.addWidget(self.getPositionsButton)

        self.finalizePositionsButton = QPushButton(self.horizontalLayoutWidget)
        self.finalizePositionsButton.setObjectName(u"finalizePositionsButton")

        self.positionsButtonsLayout.addWidget(self.finalizePositionsButton)

        self.resetPositionsButton = QPushButton(self.horizontalLayoutWidget)
        self.resetPositionsButton.setObjectName(u"resetPositionsButton")

        self.positionsButtonsLayout.addWidget(self.resetPositionsButton)

        self.SlowPositionsLabel = QLabel(self.centralwidget)
        self.SlowPositionsLabel.setObjectName(u"SlowPositionsLabel")
        self.SlowPositionsLabel.setGeometry(QRect(390, 20, 171, 19))
        self.presetsBox = QGroupBox(self.centralwidget)
        self.presetsBox.setObjectName(u"presetsBox")
        self.presetsBox.setGeometry(QRect(30, 370, 621, 251))
        self.horizontalLayoutWidget_3 = QWidget(self.presetsBox)
        self.horizontalLayoutWidget_3.setObjectName(u"horizontalLayoutWidget_3")
        self.horizontalLayoutWidget_3.setGeometry(QRect(10, 190, 281, 41))
        self.presetsLayout = QHBoxLayout(self.horizontalLayoutWidget_3)
        self.presetsLayout.setObjectName(u"presetsLayout")
        self.presetsLayout.setContentsMargins(0, 0, 0, 0)
        self.addPresetButton = QPushButton(self.horizontalLayoutWidget_3)
        self.addPresetButton.setObjectName(u"addPresetButton")

        self.presetsLayout.addWidget(self.addPresetButton)

        self.removePresetButton = QPushButton(self.horizontalLayoutWidget_3)
        self.removePresetButton.setObjectName(u"removePresetButton")

        self.presetsLayout.addWidget(self.removePresetButton)

        self.channelExposureList = QListWidget(self.presetsBox)
        self.channelExposureList.setObjectName(u"channelExposureList")
        self.channelExposureList.setGeometry(QRect(370, 20, 231, 171))
        self.formLayoutWidget = QWidget(self.presetsBox)
        self.formLayoutWidget.setObjectName(u"formLayoutWidget")
        self.formLayoutWidget.setGeometry(QRect(10, 30, 311, 141))
        self.channelExposureLayout = QFormLayout(self.formLayoutWidget)
        self.channelExposureLayout.setObjectName(u"channelExposureLayout")
        self.channelExposureLayout.setContentsMargins(0, 0, 0, 0)
        self.channelLabel = QLabel(self.formLayoutWidget)
        self.channelLabel.setObjectName(u"channelLabel")

        self.channelExposureLayout.setWidget(0, QFormLayout.LabelRole, self.channelLabel)

        self.presets = QComboBox(self.formLayoutWidget)
        self.presets.setObjectName(u"presets")

        self.channelExposureLayout.setWidget(0, QFormLayout.FieldRole, self.presets)

        self.exposureLabel = QLabel(self.formLayoutWidget)
        self.exposureLabel.setObjectName(u"exposureLabel")

        self.channelExposureLayout.setWidget(1, QFormLayout.LabelRole, self.exposureLabel)

        self.exposure = QLineEdit(self.formLayoutWidget)
        self.exposure.setObjectName(u"exposure")

        self.channelExposureLayout.setWidget(1, QFormLayout.FieldRole, self.exposure)

        self.timeIntervalLabel = QLabel(self.formLayoutWidget)
        self.timeIntervalLabel.setObjectName(u"timeIntervalLabel")

        self.channelExposureLayout.setWidget(2, QFormLayout.LabelRole, self.timeIntervalLabel)

        self.minTimeIntervalSpinBox = QSpinBox(self.formLayoutWidget)
        self.minTimeIntervalSpinBox.setObjectName(u"minTimeIntervalSpinBox")
        self.minTimeIntervalSpinBox.setMaximum(10000)

        self.channelExposureLayout.setWidget(2, QFormLayout.FieldRole, self.minTimeIntervalSpinBox)

        self.nTimePointsLabel = QLabel(self.formLayoutWidget)
        self.nTimePointsLabel.setObjectName(u"nTimePointsLabel")

        self.channelExposureLayout.setWidget(3, QFormLayout.LabelRole, self.nTimePointsLabel)

        self.nTimePointsSpinBox = QSpinBox(self.formLayoutWidget)
        self.nTimePointsSpinBox.setObjectName(u"nTimePointsSpinBox")
        self.nTimePointsSpinBox.setMaximum(500)
        self.nTimePointsSpinBox.setValue(1)

        self.channelExposureLayout.setWidget(3, QFormLayout.FieldRole, self.nTimePointsSpinBox)

        self.label = QLabel(self.presetsBox)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(370, 200, 231, 41))
        self.horizontalLayoutWidget_2 = QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setObjectName(u"horizontalLayoutWidget_2")
        self.horizontalLayoutWidget_2.setGeometry(QRect(30, 50, 616, 194))
        self.positionListsLayout = QHBoxLayout(self.horizontalLayoutWidget_2)
        self.positionListsLayout.setObjectName(u"positionListsLayout")
        self.positionListsLayout.setContentsMargins(0, 0, 0, 0)
        self.fastPositions = QListWidget(self.horizontalLayoutWidget_2)
        self.fastPositions.setObjectName(u"fastPositions")

        self.positionListsLayout.addWidget(self.fastPositions)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.sendToSlowButton = QPushButton(self.horizontalLayoutWidget_2)
        self.sendToSlowButton.setObjectName(u"sendToSlowButton")

        self.verticalLayout.addWidget(self.sendToSlowButton)

        self.sendToFastButton = QPushButton(self.horizontalLayoutWidget_2)
        self.sendToFastButton.setObjectName(u"sendToFastButton")

        self.verticalLayout.addWidget(self.sendToFastButton)


        self.positionListsLayout.addLayout(self.verticalLayout)

        self.slowPositions = QListWidget(self.horizontalLayoutWidget_2)
        self.slowPositions.setObjectName(u"slowPositions")

        self.positionListsLayout.addWidget(self.slowPositions)

        self.horizontalLayoutWidget_4 = QWidget(self.centralwidget)
        self.horizontalLayoutWidget_4.setObjectName(u"horizontalLayoutWidget_4")
        self.horizontalLayoutWidget_4.setGeometry(QRect(320, 640, 318, 51))
        self.eventButtonsLayout = QHBoxLayout(self.horizontalLayoutWidget_4)
        self.eventButtonsLayout.setObjectName(u"eventButtonsLayout")
        self.eventButtonsLayout.setContentsMargins(0, 0, 0, 0)
        self.constructEventsButton = QPushButton(self.horizontalLayoutWidget_4)
        self.constructEventsButton.setObjectName(u"constructEventsButton")

        self.eventButtonsLayout.addWidget(self.constructEventsButton)

        self.resetEventsButton = QPushButton(self.horizontalLayoutWidget_4)
        self.resetEventsButton.setObjectName(u"resetEventsButton")

        self.eventButtonsLayout.addWidget(self.resetEventsButton)

        self.closeWindowButton = QPushButton(self.horizontalLayoutWidget_4)
        self.closeWindowButton.setObjectName(u"closeWindowButton")

        self.eventButtonsLayout.addWidget(self.closeWindowButton)

        EventsWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(EventsWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 685, 21))
        EventsWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(EventsWindow)
        self.statusbar.setObjectName(u"statusbar")
        EventsWindow.setStatusBar(self.statusbar)

        self.retranslateUi(EventsWindow)

        QMetaObject.connectSlotsByName(EventsWindow)
    # setupUi

    def retranslateUi(self, EventsWindow):
        EventsWindow.setWindowTitle(QCoreApplication.translate("EventsWindow", u"MainWindow", None))
        self.FastPositionsLabel.setText(QCoreApplication.translate("EventsWindow", u"Fast Positions in Phase", None))
        self.getPositionsButton.setText(QCoreApplication.translate("EventsWindow", u"Get Positions", None))
        self.finalizePositionsButton.setText(QCoreApplication.translate("EventsWindow", u"Finalize Positions", None))
        self.resetPositionsButton.setText(QCoreApplication.translate("EventsWindow", u"Reset Positions", None))
        self.SlowPositionsLabel.setText(QCoreApplication.translate("EventsWindow", u"Slow Positions in Phase", None))
        self.presetsBox.setTitle(QCoreApplication.translate("EventsWindow", u"Presets and Channels", None))
        self.addPresetButton.setText(QCoreApplication.translate("EventsWindow", u"Add", None))
        self.removePresetButton.setText(QCoreApplication.translate("EventsWindow", u"Remove", None))
        self.channelLabel.setText(QCoreApplication.translate("EventsWindow", u"Channel:", None))
        self.exposureLabel.setText(QCoreApplication.translate("EventsWindow", u"Exposure(ms):", None))
        self.timeIntervalLabel.setText(QCoreApplication.translate("EventsWindow", u"Time Interval(s)", None))
        self.nTimePointsLabel.setText(QCoreApplication.translate("EventsWindow", u"No of TimePoints", None))
        self.label.setText(QCoreApplication.translate("EventsWindow", u"Always add phase preset First", None))
        self.sendToSlowButton.setText(QCoreApplication.translate("EventsWindow", u">", None))
        self.sendToFastButton.setText(QCoreApplication.translate("EventsWindow", u"<", None))
        self.constructEventsButton.setText(QCoreApplication.translate("EventsWindow", u"Construct Events", None))
        self.resetEventsButton.setText(QCoreApplication.translate("EventsWindow", u"Reset Events", None))
        self.closeWindowButton.setText(QCoreApplication.translate("EventsWindow", u"Close", None))
    # retranslateUi

