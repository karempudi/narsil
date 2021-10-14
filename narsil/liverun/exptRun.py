import multiprocessing as mp
import torch.multiprocessing as tmp
import psycopg2 as pgdatabase
import torch
import argparse
import os
import sys
import time
from pathlib import Path
from functools import partial
from torchvision import transforms, utils
from queue import Empty
from pycromanager import Acquisition
from narsil.liverun.utils import queueDataset, resizeOneImage, tensorizeOneImage
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from narsil.segmentation.network import basicUnet, smallerUnet

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

        self.acquireProcess = None
        self.segmentProcess = None
        self.deadAliveProcess = None

        self.acquireKillEvent = tmp.Event()
        self.segmentKillEvent = tmp.Event()
        self.deadaliveKilEvent = tmp.Event()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # datasets: These are wrappers around torch multiprocessing queues, that are used
        # to fetch data using iterable dataloader. Dataloader
        self.segmentDataset = queueDataset(self.segmentQueue) 

        self.cellSegNet = None
        self.channelSegNet = None
    
    def createProcesses(self):
        # all the stuff needed to for processing functions
        # like the networks used etc
        self.acquireProcess = tmp.Process(target=self.acquire, name='acquireProcess')
        self.segmentProcess = tmp.Process(target=self.segment, name='segmentProcess')
        self.deadAliveProcess = tmp.Process(target=self.deadalive, name='deadaliveProcess')


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
        self.tensorize = tensorizeOneImage()
        self.segTransforms = transforms.Compose([self.resize, self.tensorize])


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

    def waitForPFS(self, ):
        pass

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
                        sys.stdout.write(f"Image shape segmented: {image.shape}--{data['position']} -- {data['time']} \n")
                        sys.stdout.flush()
                        if image == None:
                            time.sleep(2)

                        # segment here and cut channels and write the data to disk
                        cellSegMask = self.cellSegNet(image)
                        channelSegMask = self.channelSegNet(image)
                        locations = self.findLocations(channelSegMask)

                        # Keep track of locaitons, barcodes in each image and stuff needed to go back and map


                        self.recordInDatabase('segment', {'time': data['time'], 'position': data['position']})

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
        # doesn't get copied as it is not picklable.
        self.segmentProcess.start()
    
    def stop(self):
        self.segmentKillEvent.set()
        self.acquireKillEvent.set()
    
class tweezerWindow(QMainWindow):

    def __init__(self):
        pass


if __name__ == "__main__":
    print("Experiment Processes launch ...")
    # parse the argments and create appropriate processes and queues