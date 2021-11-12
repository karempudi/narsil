# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'liveWindow.ui'
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
from PySide6.QtWidgets import (QApplication, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMenuBar, QPushButton,
    QScrollBar, QSizePolicy, QStatusBar, QWidget)

from pyqtgraph import ImageView

class Ui_LiveWindow(object):
    def setupUi(self, LiveWindow):
        if not LiveWindow.objectName():
            LiveWindow.setObjectName(u"LiveWindow")
        LiveWindow.resize(955, 616)
        self.centralwidget = QWidget(LiveWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.highlightBox = QGroupBox(self.centralwidget)
        self.highlightBox.setObjectName(u"highlightBox")
        self.highlightBox.setGeometry(QRect(230, 10, 531, 61))
        self.horizontalLayoutWidget = QWidget(self.highlightBox)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(10, 30, 511, 27))
        self.horizontalLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.channelNoLabel = QLabel(self.horizontalLayoutWidget)
        self.channelNoLabel.setObjectName(u"channelNoLabel")

        self.horizontalLayout.addWidget(self.channelNoLabel)

        self.channelNoText = QLineEdit(self.horizontalLayoutWidget)
        self.channelNoText.setObjectName(u"channelNoText")

        self.horizontalLayout.addWidget(self.channelNoText)

        self.BlockNoLabel = QLabel(self.horizontalLayoutWidget)
        self.BlockNoLabel.setObjectName(u"BlockNoLabel")

        self.horizontalLayout.addWidget(self.BlockNoLabel)

        self.blockNoText = QLineEdit(self.horizontalLayoutWidget)
        self.blockNoText.setObjectName(u"blockNoText")

        self.horizontalLayout.addWidget(self.blockNoText)

        self.pushButton_2 = QPushButton(self.horizontalLayoutWidget)
        self.pushButton_2.setObjectName(u"pushButton_2")

        self.horizontalLayout.addWidget(self.pushButton_2)

        self.controlsBox = QGroupBox(self.centralwidget)
        self.controlsBox.setObjectName(u"controlsBox")
        self.controlsBox.setGeometry(QRect(10, 10, 221, 61))
        self.horizontalLayoutWidget_2 = QWidget(self.controlsBox)
        self.horizontalLayoutWidget_2.setObjectName(u"horizontalLayoutWidget_2")
        self.horizontalLayoutWidget_2.setGeometry(QRect(0, 20, 221, 41))
        self.horizontalLayout_2 = QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.startImaging = QPushButton(self.horizontalLayoutWidget_2)
        self.startImaging.setObjectName(u"startImaging")

        self.horizontalLayout_2.addWidget(self.startImaging)

        self.stopImaging = QPushButton(self.horizontalLayoutWidget_2)
        self.stopImaging.setObjectName(u"stopImaging")

        self.horizontalLayout_2.addWidget(self.stopImaging)

        self.liveImageGraphics = ImageView(self.centralwidget)
        self.liveImageGraphics.setObjectName(u"liveImageGraphics")
        self.liveImageGraphics.setGeometry(QRect(10, 120, 921, 331))
        self.horizontalScrollBar = QScrollBar(self.centralwidget)
        self.horizontalScrollBar.setObjectName(u"horizontalScrollBar")
        self.horizontalScrollBar.setGeometry(QRect(360, 460, 160, 16))
        self.horizontalScrollBar.setOrientation(Qt.Horizontal)
        LiveWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(LiveWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 955, 22))
        LiveWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(LiveWindow)
        self.statusbar.setObjectName(u"statusbar")
        LiveWindow.setStatusBar(self.statusbar)

        self.retranslateUi(LiveWindow)

        QMetaObject.connectSlotsByName(LiveWindow)
    # setupUi

    def retranslateUi(self, LiveWindow):
        LiveWindow.setWindowTitle(QCoreApplication.translate("LiveWindow", u"MainWindow", None))
        self.highlightBox.setTitle(QCoreApplication.translate("LiveWindow", u"Highlight", None))
        self.channelNoLabel.setText(QCoreApplication.translate("LiveWindow", u"Channel No", None))
        self.BlockNoLabel.setText(QCoreApplication.translate("LiveWindow", u"Block No", None))
        self.pushButton_2.setText(QCoreApplication.translate("LiveWindow", u"Highlight", None))
        self.controlsBox.setTitle(QCoreApplication.translate("LiveWindow", u"Controls", None))
        self.startImaging.setText(QCoreApplication.translate("LiveWindow", u"Start", None))
        self.stopImaging.setText(QCoreApplication.translate("LiveWindow", u"Stop", None))
    # retranslateUi

