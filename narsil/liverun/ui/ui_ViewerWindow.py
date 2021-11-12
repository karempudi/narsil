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
from PySide6.QtWidgets import (QApplication, QGroupBox, QLabel, QLineEdit,
    QListView, QMainWindow, QMenuBar, QPushButton,
    QSizePolicy, QStatusBar, QWidget)

from pyqtgraph import ImageView

class Ui_ViewerWindow(object):
    def setupUi(self, ViewerWindow):
        if not ViewerWindow.objectName():
            ViewerWindow.setObjectName(u"ViewerWindow")
        ViewerWindow.resize(1260, 758)
        self.centralwidget = QWidget(ViewerWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(20, 20, 71, 21))
        self.positionNoLine = QLineEdit(self.centralwidget)
        self.positionNoLine.setObjectName(u"positionNoLine")
        self.positionNoLine.setGeometry(QRect(90, 20, 113, 25))
        self.activePositions = QListView(self.centralwidget)
        self.activePositions.setObjectName(u"activePositions")
        self.activePositions.setGeometry(QRect(970, 80, 256, 351))
        self.channelNoLabel = QLabel(self.centralwidget)
        self.channelNoLabel.setObjectName(u"channelNoLabel")
        self.channelNoLabel.setGeometry(QRect(240, 20, 91, 17))
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
        self.graphicsView = ImageView(self.centralwidget)
        self.graphicsView.setObjectName(u"graphicsView")
        self.graphicsView.setGeometry(QRect(40, 80, 551, 591))
        self.graphicsView.setInteractive(True)
        self.filterParametersBox = QGroupBox(self.centralwidget)
        self.filterParametersBox.setObjectName(u"filterParametersBox")
        self.filterParametersBox.setGeometry(QRect(600, 80, 341, 401))
        self.channelNoLine = QLineEdit(self.centralwidget)
        self.channelNoLine.setObjectName(u"channelNoLine")
        self.channelNoLine.setGeometry(QRect(330, 20, 113, 25))
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(1090, 530, 89, 25))
        self.fetchButton = QPushButton(self.centralwidget)
        self.fetchButton.setObjectName(u"fetchButton")
        self.fetchButton.setGeometry(QRect(490, 20, 89, 25))
        self.pushButton_2 = QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(980, 630, 191, 25))
        ViewerWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(ViewerWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1260, 22))
        ViewerWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(ViewerWindow)
        self.statusbar.setObjectName(u"statusbar")
        ViewerWindow.setStatusBar(self.statusbar)

        self.retranslateUi(ViewerWindow)

        QMetaObject.connectSlotsByName(ViewerWindow)
    # setupUi

    def retranslateUi(self, ViewerWindow):
        ViewerWindow.setWindowTitle(QCoreApplication.translate("ViewerWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("ViewerWindow", u"Position", None))
        self.channelNoLabel.setText(QCoreApplication.translate("ViewerWindow", u"Channel No", None))
        self.removeButton.setText(QCoreApplication.translate("ViewerWindow", u"Remove", None))
        self.showButton.setText(QCoreApplication.translate("ViewerWindow", u"Show", None))
        self.undoButton.setText(QCoreApplication.translate("ViewerWindow", u"Undo", None))
        self.resetButton.setText(QCoreApplication.translate("ViewerWindow", u"Reset", None))
        self.filterParametersBox.setTitle(QCoreApplication.translate("ViewerWindow", u"Filter Parameters", None))
        self.pushButton.setText(QCoreApplication.translate("ViewerWindow", u"Next Auto", None))
        self.fetchButton.setText(QCoreApplication.translate("ViewerWindow", u"Fetch", None))
        self.pushButton_2.setText(QCoreApplication.translate("ViewerWindow", u"Send Tweeze Positions", None))
    # retranslateUi

