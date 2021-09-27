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


"""
ExptProcess class that creates runs all the processes and
manages shared objects between processes and status of each process
"""
class exptRun(object):

    def __init__(self):

        # Image acquisition events that you get from GUI
        self.acquireEvents = None

        # Image process parameters needed to be set
        self.imageProcessParameters = None

        # DB parameters that you get from GUI, used for writing data
        # in to the database
        self.dbParameters = None

        # queues and kill events
        self.segmentQueue = tmp.Queue()
        self.deadaliveQueue = tmp.Queue()

        self.acquireKillEvent = tmp.Event()
        self.segmentKillEvent = tmp.Event()
        self.deadaliveKilEvent = tmp.Event()

        self.acquireProcess = tmp.Process(target=self.acquire, name='acquireProcess')
        self.segmentProcess = tmp.Process(target=self.segment, name='segmentProcess')
        self.deadAliveProcess = tmp.Process(target=self.deadalive, name='deadaliveProcess')

        # all the stuff needed to for processing functions
        # like the networks used etc
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # datasets: These are wrappers around torch multiprocessing queues, that are used
        # to fetch data using iterable dataloader. Dataloader
        self.segmentDataset = queueDataset(self.segmentQueue) 

        self.loadNets()

    def loadNets(self):
        pass


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
        con = None
        try:
            con = pgdatabase.connect(database=self.dbParameters['dbname'],
                                     user=self.dbParameters['dbuser'],
                                     password=self.dbParameters['dbpassword'])
            cur = con.cursor()
            con.autocommit = True

            # insert the arrival of the image into the database table arrival
            cur.execute("""INSERT INTO arrival (time, position, timepoint)
                        VALUES (%s, %s, %s)""", (datetime.now(), int(metadata['Axes']['position']),
                        int(metadata['Axes']['time']),))
        except pgdatabase.DatabaseError as e:
            sys.stderr.write(f"Error in writing to arrival: {e}")
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
        pass
    
    def deadalive(self):
        pass

    def growth(self):
        pass

    # basically start all the experiment processes and run
    # until the abort buttons are pressed
    def run(self):
        
        self.acquireProcess.start()
    
    def stop(self):

        self.acquireKillEvent.set()


if __name__ == "__main__":
    print("Experiment Processes launch ...")

    # parse the argments and create appropriate processes and queues








