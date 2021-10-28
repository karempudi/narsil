import multiprocessing as mp
import torch.multiprocessing as tmp
import psycopg2 as pgdatabase
import torch
import argparse
import os
import sys
import time
import math
from pathlib import Path
from functools import partial
from torchvision import transforms, utils
from queue import Empty
from pycromanager import Acquisition
from narsil.liverun.utils import queueDataset, resizeOneImage, tensorizeOneImage, normalize
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from narsil.segmentation.network import basicUnet, smallerUnet
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from skimage import io
from datetime import datetime
from scipy.signal import find_peaks
import numpy as np

"""
ExptProcess class that creates runs all the processes and
manages shared objects between processes and status of each process
"""
class exptRun(object):

    def __init__(self):

        # Image acquisition events that you get from GUI
        self.acquireEvents = None

        # Image process parameters needed to be set
        # network model paths are also in imageProcessParameter
        self.imageProcessParameters = None

        # DB parameters that you get from GUI, used for writing data
        # in to the database
        self.dbParameters = None

        # queues and kill events
        self.segmentQueue = tmp.Queue()
        self.deadaliveQueue = tmp.Queue()

        #self.acquireProcess = None
        #self.segmentProcess = None
        #self.deadAliveProcess = None

        self.acquireKillEvent = tmp.Event()
        self.segmentKillEvent = tmp.Event()
        self.deadaliveKilEvent = tmp.Event()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # datasets: These are wrappers around torch multiprocessing queues, that are used
        # to fetch data using iterable dataloader. Dataloader
        self.segmentDataset = queueDataset(self.segmentQueue) 

        self.cellSegNet = None
        self.channelSegNet = None

        self.channelProcessParameters = {
            'segThreshold': 0.9,
            'minPeaksDistance': 25,
            'barcodeWidth': 48,
            'minChannelLength':100,
            'rowThreshold': 10000,
            'channelRowLocations': [400, 1200],
            'barcodeLength': 800
        }
    
    #def createProcesses(self):
    #    # all the stuff needed to for processing functions
    #    # like the networks used etc
    #    self.acquireProcess = tmp.Process(target=self.acquire, name='acquireProcess')
    #    self.segmentProcess = tmp.Process(target=self.segment, name='segmentProcess')
    #    self.deadAliveProcess = tmp.Process(target=self.deadalive, name='deadaliveProcess')


    def loadNets(self):
        sys.stdout.write(f"Loading networks and sending to device ...\n")
        sys.stdout.flush()

        # Load the cell-segmentation and channel-segmentation model
        cellSegModelPath = Path(self.imageProcessParameters["cellModelPath"])
        cellNetState = torch.load(cellSegModelPath, map_location=self.device)
            # use the net depending on what model is loaded
        if cellNetState['modelParameters']['netType'] == 'big':
            self.cellSegNet = basicUnet(cellNetState['modelParameters']['transposeConv'])
        elif cellNetState['modelParameters']['netType'] == 'small':
            self.cellSegNet = smallerUnet(cellNetState['modelParameters']['transposeConv'])
        
        self.cellSegNet.load_state_dict(cellNetState['model_state_dict'])
        self.cellSegNet.eval()

        channelSegModelPath = Path(self.imageProcessParameters["channelModelPath"])
        channelNetState = torch.load(channelSegModelPath, map_location=self.device)
            # use the net depending on what model is loaded
        if channelNetState['modelParameters']['netType'] == 'big':
            self.channelSegNet = basicUnet(channelNetState['modelParameters']['transposeConv'])
        elif channelNetState['modelParameters']['netType'] == 'small':
            self.channelSegNet = smallerUnet(channelNetState['modelParameters']['transposeConv'])

        self.channelSegNet.load_state_dict(channelNetState['model_state_dict'])
        self.channelSegNet.eval()

        sys.stdout.write(f"Networks loaded onto {self.device} successfully ...\n")
        sys.stdout.flush()

    

    # all the transformations can be set depending on the images taken
    def setImageTransforms(self):
        # operations on the images before processing
        self.phaseImageSize = (self.imageProcessParameters["imageHeight"], self.imageProcessParameters["imageWidth"])
        self.resize = resizeOneImage(self.phaseImageSize, self.phaseImageSize) 
        self.normalize = normalize() 
        self.tensorize = tensorizeOneImage()
        self.segTransforms = transforms.Compose([self.resize, self.normalize, self.tensorize])


    def putImagesInSegQueue(self, image, metadata):
        sys.stdout.write(f"Image Acquired ... {image.shape} .. {metadata['Axes']} .. {metadata['Time']}\n")
        sys.stdout.flush()
        # transform the image into a tensor
        imageTensor = self.segTransforms(image)

        # put the image into the segmentDataset
        try:
            self.segmentQueue.put({'image': imageTensor,
                                    'position': metadata['Axes']['position'],
                                    'time': metadata['Axes']['time']})
        except Exception as error:
            sys.stderr.write(f"Image at position: {metadata['Axes']['position']} and time: {metadata['Axes']['time']}\n")
            sys.stderr.write(f"Error: {error}")
            sys.stderr.flush()

        # write to database
        self.recordInDatabase('arrival', metadata)

    def recordInDatabase(self, tableName, data):
        con = None
        try:
            con = pgdatabase.connect(database=self.dbParameters['dbname'],
                                    user=self.dbParameters['dbuser'],
                                    password=self.dbParameters['dbpassword'])
            cur = con.cursor()
            con.autocommit = True

            if tableName == 'arrival':
                # insert the arrival of the image into the database table arrival
                cur.execute("""INSERT INTO arrival (time, position, timepoint)
                            VALUES (%s, %s, %s)""", (datetime.now(), int(data['Axes']['position']),
                            int(data['Axes']['time']),))
            elif tableName == 'segment':
                cur.execute("""INSERT INTO segment (time, position, timepoint)
                            VALUES (%s, %s, %s)""", (datetime.now(), int(data['position']),
                            int(data['time']),))

        except pgdatabase.DatabaseError as e:
            sys.stderr.write(f"Error in writing to database: {e}")
            sys.stderr.flush()
        finally:
            if con:
                con.close()

    def waitForPFS(self, event, bridge, event_queue):
        # wait for focus before acquisition 
        core = bridge.get_core()
        core.full_focus()
        return event

    # fake acquiring outside to test positions  
    def acquireFake(self):

        testDataDir = Path("C:\\Users\\Praneeth\\Documents\\Elflab\\Code\\testdata\\hetero40x")
        for event in self.acquireEvents:
            print(f"{event['axes']['position']} -- {event['axes']['time']}")
            positionStr = "Pos10" + str(event['axes']['position'])
            imgName = imgFilenameFromNumber(int(event['axes']['time']))
            channelName = str(event['channel']['config'])
            imagePath = testDataDir / positionStr / channelName/ imgName
            #print(event)
            metadata  = {
                'Axes': {'position': int(event['axes']['position']), 
                         'time' : int(event['axes']['time'])},
                'Time': str(datetime.now())
            }
            img = io.imread(imagePath)
            self.putImagesInSegQueue(img, metadata)
            print(imagePath)
            print("--------")

            time.sleep(3)

        while not self.acquireKillEvent.is_set():
            try:
                time.sleep(2)
            except KeyboardInterrupt:
                self.acquireKillEvent.set()
                sys.stdout.write("AcquireFake process interrupted using keyboard\n")
                sys.stdout.flush()
        
        sys.stdout.write("AcquireFake process completed successfully")
        sys.stdout.flush()

    def acquire(self):

        with Acquisition(image_process_fn=partial(self.putImagesInSegQueue), debug=False) as acq:
            acq.acquire(self.acquireEvents)

        while not self.acquireKillEvent.is_set():
            try:
                time.sleep(2)
            except KeyboardInterrupt:
                self.acquireKillEvent.set()
                sys.stdout.write("Acquire process interrupted using keyboard\n")
                sys.stdout.flush()

        sys.stdout.write("Acquire process completed successfully\n")
        sys.stdout.flush()

    # do all the writing to file system using this function,
    # abstract out the logic for different cases 
    def writeFile(self, image, type, position, time, channelNo=None):
        pass



    # return the number of channels detected, locations to write to database 
    # assume it is one image per batch
    # TODO: batching of images done later
    def processChannels(self, image, position, time):

        # pass through net and get the results
        channelSegMask = torch.sigmoid(self.channelSegNet(image)) > self.channelProcessParameters['segThreshold']

        # sent to cpu and saved according to position and timepoint
        channelSegMaskCpu = channelSegMask.cpu().detach().numpy().squeeze(0).squeeze(0) * 255.0
        channelMaskFilename = str(position) + "_" + str(time) + ".tiff"
        channelMaskFilename = Path(self.imageProcessParameters["saveDir"]) / channelMaskFilename
        io.imsave(channelMaskFilename, channelSegMaskCpu.astype('uint8'), compress=6, check_contrast=False,
                    plugin='tifffile')


        # grab locations of barcode # will be used for checking with 100x images later
        hist = np.sum(channelSegMaskCpu, axis = 0) > self.channelProcessParameters['minChannelLength']
        peaks, _ = find_peaks(hist, distance = self.channelProcessParameters['minPeaksDistance'])
        indices_with_larger_gaps = np.where(np.ediff1d(peaks) > self.channelProcessParameters['barcodeWidth'])[0]

        # there are locaitons of the channel before and after the the gap
        locations_before_barcode = peaks[indices_with_larger_gaps]
        locations_after_barcode = peaks[indices_with_larger_gaps + 1]

        # take the mean of the locations to get the center, convert to int (from uint) for the next steps
        locations_barcode = np.rint(np.mean((locations_before_barcode,
                         locations_after_barcode), axis=0)).astype('int')

        sum_rows = np.sum(channelSegMaskCpu, axis = 1).astype('int')
        row_locations = np.argwhere(np.diff(np.sign(sum_rows - self.channelProcessParameters['rowThreshold']))).flatten()

        if len(row_locations) != 2:
            row_x1 = self.channelProcessParameters['channelRowLocations'][0]
            row_x2 = self.channelProcessParameters['channelRowLocations'][1]
        else:
            row_x1 = row_locations[0]
            row_x2 = row_x1 + self.channelProcessParameters['barcodeLength']

        # grab barcode and then grab the channels in each image and write
        barcodeImages = []
        phase_img = image.cpu().detach().numpy().squeeze(0).squeeze(0)
        barcodeWidth = self.channelProcessParameters['barcodeWidth']
        for location in locations_barcode:
            barcode_img = image[row_x1:row_x1, location - barcodeWidth//2: location + barcodeWidth//2]
            barcodeImages.append(barcode_img)
        
        # grab channels between barcodes and record the info in database
        # TODO:


        sys.stdout.write(str(channelMaskFilename) + "\n")
        sys.stdout.write(f"No of barcode regions detected: {len(barcodeImages)}\n")
        sys.stdout.flush()

    def processCells(self, image, position, time):
        pass
    
    
    def segment(self):
        # segmentation loop for both cell and channels
        sys.stdout.write(f"Starting segmentation ... \n")
        sys.stdout.flush()

        while not self.segmentKillEvent.is_set():
            try:
                dataloader = DataLoader(self.segmentDataset, batch_size=1)
                with torch.no_grad():
                    for data in dataloader:
                        image = data['image'].to(self.device)
                        if image == None:
                            time.sleep(2)
                        self.processChannels(image, int(data['position']), int(data['time']))

                        self.recordInDatabase('segment', {'time': data['time'], 'position': data['position']})
                        sys.stdout.write(f"Image shape segmented: {image.shape}--{data['position']} -- {data['time']} \n")
                        sys.stdout.flush()

            except Empty:
                sys.stdout.write("Segmentation queue is empty .. but process shutdown is not happening\n")
                sys.stdout.flush()
            except KeyboardInterrupt:
                self.segmentKillEvent.set()
                sys.stdout.write(f"Segmetation process interrupted using keyboard\n")
                sys.stdout.flush()

        sys.stdout.write("Segmentation process completed successfully\n")
        sys.stdout.flush()

    def findLocations(self, channelMask):
        pass
    
    def deadalive(self):
        pass

    def growth(self):
        pass

    # basically start all the experiment processes and run
    # until the abort buttons are pressed
    def run(self):
        self.loadNets()
        self.createProcesses()
        self.acquireProcess.start()
        self.acquireProcess = None # set this to none so that the process context
        # doesn't get copied as it is not picklable
        self.segmentProcess.start()
    
    def stop(self):
        self.segmentKillEvent.set()
        self.acquireKillEvent.set()

    # if it fails write the state and bail, and use this state to restart after adjusting 
    def savedState(self):
        pass
    
def runProcesses(exptRunObject):
    exptRunObject.loadNets()
    exptRunObject.acquireKillEvent.clear()
    acquireProcess = tmp.Process(target=exptRunObject.acquireFake, name='Acquire Process')
    acquireProcess.start()

    exptRunObject.segmentKillEvent.clear()
    segmentProcess = tmp.Process(target=exptRunObject.segment, name='Segment Process')
    segmentProcess.start()

# In the datasets image names are img_000000000.tiff format.
def imgFilenameFromNumber(number):
    if number == 0:
        num_digits = 1
    else:
        num_digits = int(math.log10(number)) + 1
    imgFilename = 'img_' + '0' * (9 - num_digits) + str(number) + '.tiff'
    return imgFilename


class tweezerWindow(QMainWindow):

    def __init__(self):
        pass

if __name__ == "__main__":
    print("Experiment Processes launch ...")
    # parse the argments and create appropriate processes and queues