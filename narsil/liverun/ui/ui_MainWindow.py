# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.2.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMenu, QMenuBar,
    QProgressBar, QPushButton, QSizePolicy, QStatusBar,
    QVBoxLayout, QWidget)

from pyqtgraph import PlotWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(914, 849)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.Setup = QGroupBox(self.centralwidget)
        self.Setup.setObjectName(u"Setup")
        self.Setup.setGeometry(QRect(10, 20, 201, 151))
        self.setupButton = QPushButton(self.Setup)
        self.setupButton.setObjectName(u"setupButton")
        self.setupButton.setGeometry(QRect(10, 30, 88, 27))
        self.viewExptSetupButton = QPushButton(self.Setup)
        self.viewExptSetupButton.setObjectName(u"viewExptSetupButton")
        self.viewExptSetupButton.setGeometry(QRect(10, 70, 88, 27))
        self.runStatus = QGroupBox(self.centralwidget)
        self.runStatus.setObjectName(u"runStatus")
        self.runStatus.setGeometry(QRect(20, 180, 861, 271))
        self.imgAcquiredPlot = PlotWidget(self.runStatus)
        self.imgAcquiredPlot.setObjectName(u"imgAcquiredPlot")
        self.imgAcquiredPlot.setGeometry(QRect(20, 70, 181, 181))
        self.imgSegPlot = PlotWidget(self.runStatus)
        self.imgSegPlot.setObjectName(u"imgSegPlot")
        self.imgSegPlot.setGeometry(QRect(230, 70, 181, 181))
        self.deadAlivePlot = PlotWidget(self.runStatus)
        self.deadAlivePlot.setObjectName(u"deadAlivePlot")
        self.deadAlivePlot.setGeometry(QRect(440, 70, 181, 181))
        self.growthRatesPlot = PlotWidget(self.runStatus)
        self.growthRatesPlot.setObjectName(u"growthRatesPlot")
        self.growthRatesPlot.setGeometry(QRect(650, 70, 181, 181))
        self.acquiredLabel = QLabel(self.runStatus)
        self.acquiredLabel.setObjectName(u"acquiredLabel")
        self.acquiredLabel.setGeometry(QRect(20, 40, 101, 16))
        font = QFont()
        font.setPointSize(10)
        self.acquiredLabel.setFont(font)
        self.acquiredLabel.setMidLineWidth(0)
        self.acquiredLabel.setTextFormat(Qt.PlainText)
        self.segmentedLabel = QLabel(self.runStatus)
        self.segmentedLabel.setObjectName(u"segmentedLabel")
        self.segmentedLabel.setGeometry(QRect(240, 40, 131, 16))
        self.segmentedLabel.setFont(font)
        self.segmentedLabel.setMidLineWidth(0)
        self.segmentedLabel.setTextFormat(Qt.PlainText)
        self.deadaliveLabel = QLabel(self.runStatus)
        self.deadaliveLabel.setObjectName(u"deadaliveLabel")
        self.deadaliveLabel.setGeometry(QRect(440, 40, 131, 16))
        self.deadaliveLabel.setFont(font)
        self.deadaliveLabel.setMidLineWidth(0)
        self.deadaliveLabel.setTextFormat(Qt.PlainText)
        self.growthRatesLabel = QLabel(self.runStatus)
        self.growthRatesLabel.setObjectName(u"growthRatesLabel")
        self.growthRatesLabel.setGeometry(QRect(650, 40, 131, 16))
        self.growthRatesLabel.setFont(font)
        self.growthRatesLabel.setMidLineWidth(0)
        self.growthRatesLabel.setTextFormat(Qt.PlainText)
        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(230, 20, 651, 151))
        self.stopExptButton = QPushButton(self.groupBox_3)
        self.stopExptButton.setObjectName(u"stopExptButton")
        self.stopExptButton.setGeometry(QRect(20, 70, 131, 27))
        self.startExptButton = QPushButton(self.groupBox_3)
        self.startExptButton.setObjectName(u"startExptButton")
        self.startExptButton.setGeometry(QRect(20, 30, 131, 27))
        self.createDbButton = QPushButton(self.groupBox_3)
        self.createDbButton.setObjectName(u"createDbButton")
        self.createDbButton.setGeometry(QRect(270, 30, 121, 27))
        self.tweezePositionsButton = QPushButton(self.groupBox_3)
        self.tweezePositionsButton.setObjectName(u"tweezePositionsButton")
        self.tweezePositionsButton.setGeometry(QRect(400, 110, 121, 27))
        self.createTablesButton = QPushButton(self.groupBox_3)
        self.createTablesButton.setObjectName(u"createTablesButton")
        self.createTablesButton.setGeometry(QRect(270, 70, 121, 27))
        self.deleteDbButton = QPushButton(self.groupBox_3)
        self.deleteDbButton.setObjectName(u"deleteDbButton")
        self.deleteDbButton.setGeometry(QRect(400, 30, 121, 27))
        self.deleteTablesButton = QPushButton(self.groupBox_3)
        self.deleteTablesButton.setObjectName(u"deleteTablesButton")
        self.deleteTablesButton.setGeometry(QRect(400, 70, 121, 27))
        self.horizontalLayoutWidget = QWidget(self.groupBox_3)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(10, 110, 250, 31))
        self.moveLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.moveLayout.setObjectName(u"moveLayout")
        self.moveLayout.setContentsMargins(0, 0, 0, 0)
        self.posNoLabel = QLabel(self.horizontalLayoutWidget)
        self.posNoLabel.setObjectName(u"posNoLabel")

        self.moveLayout.addWidget(self.posNoLabel)

        self.posNoInputBox = QLineEdit(self.horizontalLayoutWidget)
        self.posNoInputBox.setObjectName(u"posNoInputBox")

        self.moveLayout.addWidget(self.posNoInputBox)

        self.moveToPositionButton = QPushButton(self.horizontalLayoutWidget)
        self.moveToPositionButton.setObjectName(u"moveToPositionButton")

        self.moveLayout.addWidget(self.moveToPositionButton)

        self.liveButton = QPushButton(self.groupBox_3)
        self.liveButton.setObjectName(u"liveButton")
        self.liveButton.setGeometry(QRect(270, 110, 121, 27))
        self.statisticsBox = QGroupBox(self.centralwidget)
        self.statisticsBox.setObjectName(u"statisticsBox")
        self.statisticsBox.setGeometry(QRect(660, 460, 221, 301))
        self.deadAliveStatsButton = QPushButton(self.statisticsBox)
        self.deadAliveStatsButton.setObjectName(u"deadAliveStatsButton")
        self.deadAliveStatsButton.setGeometry(QRect(50, 30, 88, 27))
        self.fillRateButton = QPushButton(self.statisticsBox)
        self.fillRateButton.setObjectName(u"fillRateButton")
        self.fillRateButton.setGeometry(QRect(50, 70, 88, 27))
        self.resetButton = QPushButton(self.centralwidget)
        self.resetButton.setObjectName(u"resetButton")
        self.resetButton.setGeometry(QRect(610, 780, 75, 23))
        self.experimentProgressBox = QGroupBox(self.centralwidget)
        self.experimentProgressBox.setObjectName(u"experimentProgressBox")
        self.experimentProgressBox.setGeometry(QRect(20, 460, 631, 301))
        self.verticalLayoutWidget = QWidget(self.experimentProgressBox)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(20, 50, 191, 231))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.label_3 = QLabel(self.verticalLayoutWidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.label_3)

        self.label = QLabel(self.verticalLayoutWidget)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.label)

        self.progressBar = QProgressBar(self.verticalLayoutWidget)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setValue(24)

        self.verticalLayout.addWidget(self.progressBar)

        self.lineEdit = QLineEdit(self.verticalLayoutWidget)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setReadOnly(True)

        self.verticalLayout.addWidget(self.lineEdit)

        self.label_2 = QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.label_2)

        self.progressBar_2 = QProgressBar(self.verticalLayoutWidget)
        self.progressBar_2.setObjectName(u"progressBar_2")
        self.progressBar_2.setValue(24)

        self.verticalLayout.addWidget(self.progressBar_2)

        self.lineEdit_2 = QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_2.setObjectName(u"lineEdit_2")
        self.lineEdit_2.setReadOnly(True)

        self.verticalLayout.addWidget(self.lineEdit_2)

        self.verticalLayoutWidget_2 = QWidget(self.experimentProgressBox)
        self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(220, 50, 191, 231))
        self.verticalLayout_2 = QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.label_4 = QLabel(self.verticalLayoutWidget_2)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label_4)

        self.label_5 = QLabel(self.verticalLayoutWidget_2)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label_5)

        self.progressBar_3 = QProgressBar(self.verticalLayoutWidget_2)
        self.progressBar_3.setObjectName(u"progressBar_3")
        self.progressBar_3.setValue(24)

        self.verticalLayout_2.addWidget(self.progressBar_3)

        self.lineEdit_3 = QLineEdit(self.verticalLayoutWidget_2)
        self.lineEdit_3.setObjectName(u"lineEdit_3")
        self.lineEdit_3.setReadOnly(True)

        self.verticalLayout_2.addWidget(self.lineEdit_3)

        self.label_6 = QLabel(self.verticalLayoutWidget_2)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label_6)

        self.progressBar_4 = QProgressBar(self.verticalLayoutWidget_2)
        self.progressBar_4.setObjectName(u"progressBar_4")
        self.progressBar_4.setValue(24)

        self.verticalLayout_2.addWidget(self.progressBar_4)

        self.lineEdit_4 = QLineEdit(self.verticalLayoutWidget_2)
        self.lineEdit_4.setObjectName(u"lineEdit_4")
        self.lineEdit_4.setReadOnly(True)

        self.verticalLayout_2.addWidget(self.lineEdit_4)

        self.verticalLayoutWidget_3 = QWidget(self.experimentProgressBox)
        self.verticalLayoutWidget_3.setObjectName(u"verticalLayoutWidget_3")
        self.verticalLayoutWidget_3.setGeometry(QRect(420, 50, 191, 231))
        self.verticalLayout_3 = QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.label_7 = QLabel(self.verticalLayoutWidget_3)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.label_7)

        self.label_8 = QLabel(self.verticalLayoutWidget_3)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.label_8)

        self.progressBar_5 = QProgressBar(self.verticalLayoutWidget_3)
        self.progressBar_5.setObjectName(u"progressBar_5")
        self.progressBar_5.setValue(24)

        self.verticalLayout_3.addWidget(self.progressBar_5)

        self.lineEdit_5 = QLineEdit(self.verticalLayoutWidget_3)
        self.lineEdit_5.setObjectName(u"lineEdit_5")
        self.lineEdit_5.setReadOnly(True)

        self.verticalLayout_3.addWidget(self.lineEdit_5)

        self.label_9 = QLabel(self.verticalLayoutWidget_3)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.label_9)

        self.progressBar_6 = QProgressBar(self.verticalLayoutWidget_3)
        self.progressBar_6.setObjectName(u"progressBar_6")
        self.progressBar_6.setValue(24)

        self.verticalLayout_3.addWidget(self.progressBar_6)

        self.lineEdit_6 = QLineEdit(self.verticalLayoutWidget_3)
        self.lineEdit_6.setObjectName(u"lineEdit_6")
        self.lineEdit_6.setReadOnly(True)

        self.verticalLayout_3.addWidget(self.lineEdit_6)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 914, 22))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuExit = QMenu(self.menubar)
        self.menuExit.setObjectName(u"menuExit")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuExit.menuAction())

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.Setup.setTitle(QCoreApplication.translate("MainWindow", u"Setup", None))
        self.setupButton.setText(QCoreApplication.translate("MainWindow", u"Setup", None))
        self.viewExptSetupButton.setText(QCoreApplication.translate("MainWindow", u"View ", None))
        self.runStatus.setTitle(QCoreApplication.translate("MainWindow", u"Experiment Status", None))
        self.acquiredLabel.setText(QCoreApplication.translate("MainWindow", u"Images Acquired", None))
        self.segmentedLabel.setText(QCoreApplication.translate("MainWindow", u"Images Segmented", None))
        self.deadaliveLabel.setText(QCoreApplication.translate("MainWindow", u"Dead Alive", None))
        self.growthRatesLabel.setText(QCoreApplication.translate("MainWindow", u"Growth Rates", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Controls", None))
        self.stopExptButton.setText(QCoreApplication.translate("MainWindow", u"Stop Expt", None))
        self.startExptButton.setText(QCoreApplication.translate("MainWindow", u"Start Expt", None))
        self.createDbButton.setText(QCoreApplication.translate("MainWindow", u"Create database", None))
        self.tweezePositionsButton.setText(QCoreApplication.translate("MainWindow", u"Tweeze Positions", None))
        self.createTablesButton.setText(QCoreApplication.translate("MainWindow", u"Create tables", None))
        self.deleteDbButton.setText(QCoreApplication.translate("MainWindow", u"Delete Database", None))
        self.deleteTablesButton.setText(QCoreApplication.translate("MainWindow", u"Delete tables", None))
        self.posNoLabel.setText(QCoreApplication.translate("MainWindow", u"Pos No:", None))
        self.moveToPositionButton.setText(QCoreApplication.translate("MainWindow", u"Move To Position", None))
        self.liveButton.setText(QCoreApplication.translate("MainWindow", u"Live", None))
        self.statisticsBox.setTitle(QCoreApplication.translate("MainWindow", u"Statistics", None))
        self.deadAliveStatsButton.setText(QCoreApplication.translate("MainWindow", u"DeadAlive", None))
        self.fillRateButton.setText(QCoreApplication.translate("MainWindow", u"Fill Rate", None))
        self.resetButton.setText(QCoreApplication.translate("MainWindow", u"Reset", None))
        self.experimentProgressBox.setTitle(QCoreApplication.translate("MainWindow", u"Experiment Progess", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Acquisition", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Positions", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Time points", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Segmentation", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Positions", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Time points", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Channel Properties", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Positions", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Time points", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuExit.setTitle(QCoreApplication.translate("MainWindow", u"Exit", None))
    # retranslateUi

