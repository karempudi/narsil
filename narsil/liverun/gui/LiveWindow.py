

from enum import auto
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PySide6.QtCore import QFile, QIODevice, QTimer, Signal, Qt, QThread
from PySide6.QtUiTools import QUiLoader

from narsil.liverun.ui.ui_LiveWindow import Ui_LiveWindow
from narsil.liverun.utils import padTo32, imgFilenameFromNumber

from narsil.segmentation.network import smallerUnet, basicUnet
from pycromanager import Bridge

from pathlib import Path
from skimage import io

from skimage.feature import canny
from scipy.ndimage import binary_fill_holes, binary_erosion


import sys
import numpy as np
import torch
import random
import cv2
from datetime import datetime



class LiveImageFetch(QThread):

    dataFetched = Signal()
    
    def __init__(self, core):
        super(LiveImageFetch, self).__init__()
        self.data = None
        self.core = core

    def run(self):
        # snap an image and set the data

        try:
            exposure = self.core.get_exposure()
            #auto_shutter = self.core.get_property('Core', 'AutoShutter')
            #self.core.set_property('Core', 'AutoShutter', 0)

            self.core.snap_image()
            tagged_image = self.core.get_tagged_image()

            self.data = np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
        except Exception as e:
            sys.stdout.write(f"Live image grabbing failed\n")
            sys.stdout.flush()
            self.data = None

        self.dataFetched.emit()

    def getData(self):
        return self.data



class LiveWindow(QMainWindow):


    def __init__(self, parameters=None):
        super(LiveWindow, self).__init__()
        self.ui = Ui_LiveWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Live window")

        self.parameters = parameters

        self.acquiring = False
        self.imgAcquireThread = None
        try:
            self.bridge = Bridge()
            self.core = self.bridge.get_core()
            self.height = self.core.get_image_height()
            self.width = self.core.get_image_width()
        except:
            sys.stdout.write(f"Micromanager couldn't be connected\n")
            sys.stdout.flush()
        self.setupButtonHandlers()

        self.ui.liveImageGraphics.ui.histogram.hide()
        self.ui.liveImageGraphics.ui.roiBtn.hide()
        self.ui.liveImageGraphics.ui.menuBtn.hide()

        self.segChannels = False
        self.channelSegNet = None
        self.segCells = False
        self.cellSegNet = None
        self.device = torch.device("cpu")
        self.pad = padTo32()
        self.timer = QTimer()
        self.timer.setInterval(300)
    
    def closeEvent(self, event):
        self.stopAcquiring(True)

    def setParameters(self, parameters):
        self.parameters = parameters
        
    def setupButtonHandlers(self):

        self.ui.startImagingButton.clicked.connect(self.acquireLive)

        self.ui.stopImagingButton.clicked.connect(self.stopAcquiring)

        self.ui.cellSegCheckBox.toggled.connect(self.setCellSegNet)
        self.ui.channelSegCheckBox.toggled.connect(self.setChannelSegNet)
        self.ui.gpuCheckBox.toggled.connect(self.setGPUDevice)

    def setGPUDevice(self, buttonState):
        if buttonState == True:
            self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
            sys.stdout.write("Selected GPU option\n")
            sys.stdout.flush()

    def setCellSegNet(self, buttonState):
        self.segCells = buttonState
        sys.stdout.write(f"Loading cell seg net: {self.parameters['cellSegNetModelPath']}\n")
        sys.stdout.flush()

    def setChannelSegNet(self, buttonState):
        self.segChannels = buttonState
        sys.stdout.write(f"Loading channel seg net: {self.parameters['channelSegNetModelPath']}\n")
        sys.stdout.flush()

    def acquireLive(self):
        # grab an image every 200 ms and pipe it throught the 

        # check if button are checked and intialize the nets
        with torch.no_grad():
            if self.segCells:
                # load the cells net
                cellSegModelPath = Path(self.parameters['cellSegNetModelPath'])
                cellNetState = torch.load(cellSegModelPath, map_location = self.device)
                # use the net depending on what model is loaded
                if cellNetState['modelParameters']['netType'] == 'big':
                    self.cellSegNet = basicUnet(cellNetState['modelParameters']['transposeConv'])
                elif cellNetState['modelParameters']['netType'] == 'small':
                    self.cellSegNet = smallerUnet(cellNetState['modelParameters']['transposeConv'])
                
                self.cellSegNet.load_state_dict(cellNetState['model_state_dict'])
                self.cellSegNet.to(self.device)
                self.cellSegNet.eval()

            if self.segChannels:
                # load the channels net
                # channel segmentation model
                channelSegModelPath = Path(self.parameters["channelSegNetModelPath"])
                channelNetState = torch.load(channelSegModelPath, map_location=self.device)
                    # use the net depending on what model is loaded
                if channelNetState['modelParameters']['netType'] == 'big':
                    self.channelSegNet = basicUnet(channelNetState['modelParameters']['transposeConv'])
                elif channelNetState['modelParameters']['netType'] == 'small':
                    self.channelSegNet = smallerUnet(channelNetState['modelParameters']['transposeConv'])

                self.channelSegNet.load_state_dict(channelNetState['model_state_dict'])
                self.channelSegNet.to(self.device)
                self.channelSegNet.eval()

        sys.stdout.write(f"Nets loaded on device\n")
        sys.stdout.flush()

        self.timer.timeout.connect(self.grabImageFake)
        self.timer.start()

    def grabImageFake(self):

        try:
            randomNumber = random.randint(0, 39)
            #path = "/home/pk/Documents/realtimeData/hetero40x/Pos103/phaseFast/" + imgFilenameFromNumber(randomNumber)

            path = "/home/pk/Documents/EXP-21-BY1006/therun/Pos12/phase/" + imgFilenameFromNumber(randomNumber)
            #path = "D:\\praneeth\\hetero40x\\Pos103\\phaseFast\\" + imgFilenameFromNumber(randomNumber)
            #imageFilename =  Path("/home/pk/Documents/realtimeData/hetero40x/Pos103/phaseFast/img_000000000.tiff")
            imageFilename = Path(path)
            image = io.imread(imageFilename)
            image = self.pad(image)
            sys.stdout.write(f"Image from {imageFilename} : {image.shape} grabbed. \n")
            sys.stdout.flush()
            imgTensor = torch.from_numpy(image.astype('float32')).unsqueeze(0).unsqueeze(0).to(self.device)
            
            if self.segChannels:
                with torch.no_grad():
                    imgTensor = (imgTensor - torch.mean(imgTensor))/torch.std(imgTensor)
                    out = torch.sigmoid(self.channelSegNet(imgTensor))
                    out_cpu = out.detach().cpu().numpy().squeeze(0).squeeze(0)
                    sys.stdout.write(f"Output shape: {out_cpu.shape} --- {datetime.now()}\n")
                    sys.stdout.flush()
                self.ui.liveImageGraphics.setImage(out_cpu.T, autoLevels=True, autoRange=False)
            elif self.segCells:
                with torch.no_grad():
                    imgTensor = (imgTensor - torch.mean(imgTensor))/torch.std(imgTensor)
                    #imgTensor += torch.randn(imgTensor.shape, device=self.device)
                    out = torch.sigmoid(self.cellSegNet(imgTensor))
                    out_cpu = out.detach().cpu().numpy().squeeze(0).squeeze(0)
                    sys.stdout.write(f"Output shape: {out_cpu.shape} --- {datetime.now()}\n")
                    sys.stdout.flush()
            
                self.ui.liveImageGraphics.setImage(out_cpu.T, autoLevels=True, autoRange=False)
                
            else:
                #image = processImage(image)
                self.ui.liveImageGraphics.setImage(image.T, autoLevels=True, autoRange=False)

        except Exception as e:
            sys.stdout.write(f"Fake grabbing failed\n")
            sys.stdout.write(f"{e}\n")
            sys.stdout.flush()
            image = np.random.normal(loc=0.0, scale=1.0, size=(100, 100))
            self.ui.liveImageGraphics.setImage(image, autoLevels=True, autoRange=False)
         
        self.acquiring = True
        sys.stdout.write(f"Image aqcuiring : {self.acquiring}\n")
        sys.stdout.flush()

    def grabImage(self):
        try:
            self.imgAcquireThread = LiveImageFetch(self.core)
            self.imgAcquireThread.dataFetched.connect(self.updateImage)
            self.imgAcquireThread.start()

        except Exception as e:
            sys.stdout.write(f"Live image grabbing failed\n")
            sys.stdout.flush()
            self.data = np.random.randint(low=0, high=2, size=(self.width, self.height))
        
        self.acquiring = True
        sys.stdout.write(f"Image aqcuiring : {self.acquiring}\n")
        sys.stdout.flush()
    

    def stopAcquiring(self, clicked):
        self.acquiring = False
        sys.stdout.write(f"Image aqcuiring : {self.acquiring}\n")
        sys.stdout.flush()
        self.timer.stop()
        # delete the nets to cleanup memory.
        del self.cellSegNet
        del self.channelSegNet
        self.cellSegNet = None
        self.channelSegNet = None
        torch.cuda.empty_cache()
    
    def updateImage(self):
        sys.stdout.write("Image acquired\n")
        sys.stdout.flush()
        imgTensor = torch.from_numpy(self.imgAcquireThread.data.astype('float32')).unsqueeze(0).unsqueeze(0)
        if self.segChannels:
            with torch.no_grad():
                out = torch.sigmoid(self.channelSegNet(imgTensor)) > 0.9
                out_cpu = out.detach().numpy().squeeze(0).squeeze(0)
                sys.stdout.write(f"Output shape: {out_cpu.shape}")
                sys.stdout.flush()
        self.ui.liveImageGraphics.setImage(self.imgAcquireThread.data.T, autoLevels=True, autoRange=False)
        sys.stdout.write(f"Image plotted : {self.imgAcquireThread.data.shape}\n")
        sys.stdout.flush()

def processImage(image):

    edges = canny(image, sigma=5)
    barcodesImg = binary_fill_holes(edges)
    barcodesImg = binary_erosion(barcodesImg).astype('uint8')
    sys.stdout.write(f" --- Image Shape: {image.shape}\n")
    sys.stdout.flush()

    return barcodesImg