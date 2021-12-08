# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'runWindow.ui'
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
from PySide6.QtWidgets import (QApplication, QMainWindow, QMenuBar, QPushButton,
    QSizePolicy, QStatusBar, QWidget)

class Ui_runWindow(object):
    def setupUi(self, runWindow):
        if not runWindow.objectName():
            runWindow.setObjectName(u"runWindow")
        runWindow.resize(490, 148)
        self.centralwidget = QWidget(runWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.loadButton = QPushButton(self.centralwidget)
        self.loadButton.setObjectName(u"loadButton")
        self.loadButton.setGeometry(QRect(60, 30, 89, 25))
        self.runButton = QPushButton(self.centralwidget)
        self.runButton.setObjectName(u"runButton")
        self.runButton.setGeometry(QRect(170, 30, 89, 25))
        self.stopButton = QPushButton(self.centralwidget)
        self.stopButton.setObjectName(u"stopButton")
        self.stopButton.setGeometry(QRect(280, 30, 89, 25))
        runWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(runWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 490, 22))
        runWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(runWindow)
        self.statusbar.setObjectName(u"statusbar")
        runWindow.setStatusBar(self.statusbar)

        self.retranslateUi(runWindow)

        QMetaObject.connectSlotsByName(runWindow)
    # setupUi

    def retranslateUi(self, runWindow):
        runWindow.setWindowTitle(QCoreApplication.translate("runWindow", u"MainWindow", None))
        self.loadButton.setText(QCoreApplication.translate("runWindow", u"Load", None))
        self.runButton.setText(QCoreApplication.translate("runWindow", u"Run", None))
        self.stopButton.setText(QCoreApplication.translate("runWindow", u"Stop", None))
    # retranslateUi

