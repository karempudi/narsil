from scipy.ndimage.measurements import find_objects
import torch.multiprocessing as mp
import psycopg2 as pgdatabase
import torch
import argparse
import os
import sys
import time
import math
import pickle
import h5py
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
from skimage.morphology import remove_small_objects
import numpy as np
from skimage.measure import regionprops, label
from skimage import img_as_ubyte
import concurrent.futures

try:
    mp.set_start_method('spawn')
except:
        pass
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
        self.segmentQueue = mp.Queue()

        # write queue will grab the position and write properties and run a thread pool to parallelize 
        # the calculations as writing is the slowest part of the system
        self.writeQueue = mp.Queue()

        #self.acquireProcess = None
        #self.segmentProcess = None
        #self.deadAliveProcess = None

        self.acquireKillEvent = mp.Event()
        self.segmentKillEvent = mp.Event()
        self.writeKillEvent = mp.Event()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # datasets: These are wrappers around torch multiprocessing queues, that are used
        # to fetch data using iterable dataloader. Dataloader
        self.segmentDataset = queueDataset(self.segmentQueue) 

        self.writeDataset = queueDataset(self.writeQueue)


        self.cellSegNet = None
        self.channelSegNet = None

        self.maxTimepoints = 50


        self.channelProcessParameters = {
            'segThreshold': 0.8,
            'minPeaksDistance': 25,
            'barcodeWidth': 48,
            'minChannelLength':100,
            'smallObjectsArea': 64,
            'magnification': 40,
            'channelsPerBlock': 21,
            'channelWidth': 36,
            'plateauSize': 15,
            
        }

        self.GPUParameters = {
            'cellNetBatchSize': 1,
            'channelNetBatchSize': 1,
            'deadAliveNetBatchSize': 20,
        }

        self.cellProcessParameters = {
            'segThreshold': 0.85,
            'smallObjectsArea': 64
        }

        self.deadAliveParameters = {

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
        self.cellSegNet.to(self.device)
        self.cellSegNet.eval()

        # channel segmentation model
        channelSegModelPath = Path(self.imageProcessParameters["channelModelPath"])
        channelNetState = torch.load(channelSegModelPath, map_location=self.device)
            # use the net depending on what model is loaded
        if channelNetState['modelParameters']['netType'] == 'big':
            self.channelSegNet = basicUnet(channelNetState['modelParameters']['transposeConv'])
        elif channelNetState['modelParameters']['netType'] == 'small':
            self.channelSegNet = smallerUnet(channelNetState['modelParameters']['transposeConv'])

        self.channelSegNet.load_state_dict(channelNetState['model_state_dict'])
        self.channelSegNet.to(self.device)
        self.channelSegNet.eval()

        # dead-alive net model


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


        # set operations on the dead-alive single mother-machine channel images


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
                cur.execute("""INSERT INTO segment (time, position, timepoint, locations, numchannels)
                            VALUES (%s, %s, %s, %s, %s)""", (datetime.now(), int(data['position']),
                            int(data['time']), data['locations'], data['numchannels'],))

            elif tableName == 'growth':
                for datapoint in data:
                    cur.execute("""INSERT INTO growth (time, position, timepoint, channelno, areas, lengths, numobjects)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)""", (datetime.now(), int(datapoint['position']),
                                int(datapoint['timepoint']), datapoint['channelno'], datapoint['areas'],
                                datapoint['lengths'], datapoint['numobjects'],))

        except pgdatabase.DatabaseError as e:
            sys.stderr.write(f"Error in writing to database: {e}\n")
            sys.stderr.flush()
        finally:
            if con:
                con.close()
    
    def getLocationsFromDatabase(self, tableName, position, time):
        con = None
        try:
            con = pgdatabase.connect(database=self.dbParameters['dbname'],
                                    user=self.dbParameters['dbuser'],
                                    password=self.dbParameters['dbpassword'])
            cur = con.cursor()
            con.autocommit = True

            if tableName == 'segment':
                cur.execute("SELECT locations FROM segment WHERE position=%s AND timepoint=%s", (position, time))

                # you get a pickled bytear that needs to be converted to numpy
                rows = cur.fetchall()

                channelLocations = pickle.loads(rows[0][0])

                return channelLocations
        except pgdatabase.DatabaseError as e:
            sys.stderr.write(f"Error in getting channel locations for m database: {e}\n")
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
        #self.loadNets()
        #testDataDir = Path("C:\\Users\\Praneeth\\Documents\\Elflab\\Code\\testdata\\hetero40x")
        #testDataDir = Path("D:\\Jimmy\\EXP-21-BY1006\\therun")
        #testDataDir = Path("D:\\praneeth\\hetero40x")
        #testDataDir = Path("/home/pk/Documents/EXP-21-BY1006/therun")
        testDataDir = Path("/home/pk/Documents/realtimeData/hetero40x")
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

            time.sleep(0.3)

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
    def writeFile(self, image, imageType, position, time, channelLocations=None):
        # construct directories if they are not there
        mainAnalysisDir = Path(self.imageProcessParameters["saveDir"])
        if imageType == 'cellSegmentation':

            filename = str(time) + '.tiff'
            positionDir = str(position)
            cellMaskDir = mainAnalysisDir / positionDir / imageType
            if not cellMaskDir.exists():
                cellMaskDir.mkdir(parents=True, exist_ok=True)

            cellMaskFilename = cellMaskDir / filename

            image  = image * 255
            io.imsave(cellMaskFilename, image.astype('uint8'), compress=6, check_contrast=False,
                        plugin='tifffile')
            sys.stdout.write(str(cellMaskFilename) + " written \n")
            sys.stdout.flush()
        elif imageType == 'phaseFullImage':
            filename = str(time) + '.tiff'
            positionDir = str(position)
            phaseDir = mainAnalysisDir / positionDir/ imageType
            if not phaseDir.exists():
                phaseDir.mkdir(parents=True, exist_ok=True)
            
            phaseFilename = phaseDir/ filename

            sys.stdout.write(f"{phaseFilename} written\n")
            sys.stdout.flush()

            io.imsave(phaseFilename, image.astype('float16'), plugin='tifffile')
            
        elif imageType == 'channelSegmentation':
            # construct filename
            filename = str(time) + '.tiff'
            positionDir = str(position)
            channelMaskDir = mainAnalysisDir / positionDir / imageType 
            if not channelMaskDir.exists():
                channelMaskDir.mkdir(parents=True, exist_ok=True)

            channelMaskFilename = channelMaskDir / filename
            image = image * 255
            io.imsave(channelMaskFilename, image.astype('uint8'), compress=6, check_contrast=False,
                        plugin='tifffile')
            sys.stdout.write(str(channelMaskFilename) + " written \n")
            sys.stdout.flush()
        elif imageType == 'oneMMChannelCellSeg':
            if channelLocations == None:
                sys.stdout.write(f"Channel locations missing for Pos: {position} and time: {time}\n")
                sys.stdout.flush()
            else:
                filename = str(time) + '.tiff'
                positionDir = str(position)
                channelWidth = self.channelProcessParameters['channelWidth'] //2
                image = image * 255
                for (i, location) in enumerate(channelLocations, 0):
                    channelNo = str(i)
                    channelDir = mainAnalysisDir / positionDir / imageType / channelNo
                    if not channelDir.exists():
                        channelDir.mkdir(parents=True, exist_ok=True)
                    
                    channelImg = image[:,
                                        location - channelWidth: location + channelWidth]
                    
                    channelFileName = channelDir / filename
                    io.imsave(channelFileName, channelImg.astype('uint8'), check_contrast=False, compress=6, plugin='tifffile')
            
            sys.stdout.write(f"{len(channelLocations)} from pos: {position} and time: {time} written\n")
            sys.stdout.flush()
            
        elif imageType == 'oneMMChannelPhase':
            # check if there are locations
            if channelLocations == None:
                sys.stdout.write(f"Channel Locations missing for Pos:{position} and time:{time}\n")
                sys.stdout.flush()
            else:
                # create directories if not existing and write the stack
                filename = str(time) + '.tiff'
                positionDir = str(position)
                channelWidth = self.channelProcessParameters['channelWidth'] // 2
                for (i, location) in enumerate(channelLocations, 0):
                    channelNo = str(i)
                    channelDir  = mainAnalysisDir / positionDir/ imageType / channelNo
                    if not channelDir.exists():
                        channelDir.mkdir(parents=True, exist_ok=True)

                    channelImg = image[:,
                                        location - channelWidth: location+ channelWidth]
                    # write the image
                    channelFileName = channelDir / filename
                    io.imsave(channelFileName, channelImg, check_contrast=False, compress = 6, plugin='tifffile')

                sys.stdout.write(f"{len(channelLocations)} from pos: {position} and time: {time} written\n")
                sys.stdout.flush()
            
        elif imageType == 'barcodes':
            # you get a list of images instead of one image
            positionDir = str(position)
            barcodesDir = mainAnalysisDir / positionDir/ imageType
            if not barcodesDir.exists():
                barcodesDir.mkdir(parents=True, exist_ok=True)
            for i, oneBarcode in enumerate(image, 0):
                filename = str(time) + "_" + str(i) + '.jpg' 
                oneBarcodeFilename = barcodesDir / filename
                io.imsave(oneBarcodeFilename, oneBarcode, check_contrast=False, compress=6,
                        plugin='tifffile')
            sys.stdout.write(f"{len(image)} barcodes written to disk \n")
            sys.stdout.flush()

    def writeFileH5Py(self, image, imageType, position, time, channelLocations=None):
        mainAnalysisDir = Path(self.imageProcessParameters["saveDir"])

        if imageType == 'cellSegmentation':
            filename = str(time) + '.tiff'
            positionDir = str(position)

            cellMaskDir = mainAnalysisDir / positionDir / imageType
            if not cellMaskDir.exists():
                cellMaskDir.mkdir(parents=True, exist_ok=True)

            cellMaskFilename = cellMaskDir / filename

            image = image * 255
            io.imsave(cellMaskFilename, image.astype('uint8'), compress=6, check_contrast=False,
                        plugin='tifffile')
            sys.stdout.write(f"{cellMaskFilename} written\n")
            sys.stdout.flush()

        elif imageType == 'phaseFullImage':
            filename = str(time) + '.tiff'
            positionDir = str(position)
            phaseDir = mainAnalysisDir / positionDir/ imageType
            if not phaseDir.exists():
                phaseDir.mkdir(parents=True, exist_ok=True)
            
            phaseFilename = phaseDir/ filename

            sys.stdout.write(f"{phaseFilename} written\n")
            sys.stdout.flush()

            io.imsave(phaseFilename, image.astype('float16'), plugin='tifffile')
       
        elif imageType == 'channelSegmentation':
            filename = str(time) + '.tiff'
            positionDir = str(position)

            channelMaskDir = mainAnalysisDir / positionDir / imageType
            if not channelMaskDir.exists():
                channelMaskDir.mkdir(parents=True, exist_ok=True)

            channelMaskFilename = channelMaskDir / filename

            image = image * 255
            io.imsave(channelMaskFilename, image.astype('uint8'), compress=6, check_contrast=False,
                        plugin='tifffile')
            sys.stdout.write(f"{channelMaskFilename} written\n")
            sys.stdout.flush()

        elif imageType == 'oneMMChannelCellSeg':
            if channelLocations == None:
                sys.stdout.write(f"Channel locations missing for Pos: {position} and time: {time}\n")
                sys.stdout.flush()

            else:
                positionDir = str(position)
                writeDir = mainAnalysisDir / positionDir/ imageType
                if not writeDir.exists():
                    writeDir.mkdir(parents=True, exist_ok=True)

                height, width = image.shape
                channelWidth = self.channelProcessParameters['channelWidth'] // 2
                if time == 0:
                    # create on hdf5 stack for each of the channel locations
                    for i, location in enumerate(channelLocations, 0):
                        filename = str(i) + '.hdf5'
                        with h5py.File(writeDir / filename, 'a') as f:
                            f.create_dataset("stack", 
                                (self.maxTimepoints, height, self.channelProcessParameters['channelWidth']),
                                dtype='float16', compression='gzip')
                            f['stack'][time] = image[:, 
                                        location - channelWidth : location + channelWidth]
                else:
                    # open and write 
                    for i, location in enumerate(channelLocations, 0):
                        filename = str(i) + '.hdf5'
                        with h5py.File(writeDir/filename, 'a') as f:
                            f['stack'][time] = image[:,
                                        location - channelWidth : location + channelWidth]
                
                sys.stdout.write(f"{len(channelLocations)} from position: {position} and time: {time} written\n")
                sys.stdout.flush()

        elif imageType == 'oneMMChannelPhase':
            if channelLocations == None:
                sys.stdout.write(f"Channel locations missing for Pos: {position} and time: {time}\n")
                sys.stdout.flush()

            else:
                positionDir = str(position)
                writeDir = mainAnalysisDir / positionDir/ imageType
                if not writeDir.exists():
                    writeDir.mkdir(parents=True, exist_ok=True)

                height, width = image.shape
                channelWidth = self.channelProcessParameters['channelWidth'] // 2
                if time == 0:
                    # create on hdf5 stack for each of the channel locations
                    for i, location in enumerate(channelLocations, 0):
                        filename = str(i) + '.hdf5'
                        sys.stdout.write(f"{writeDir/filename} --- {location} -- {channelWidth}\n")
                        sys.stdout.flush()
                        with h5py.File(writeDir / filename, 'a') as f:
                            f.create_dataset("stack", 
                                (self.maxTimepoints, height, self.channelProcessParameters['channelWidth']),
                                dtype='float16', compression='gzip')
                            f['stack'][time] = image[:, 
                                        location - channelWidth : location + channelWidth]
                else:
                    # open and write 
                    for i, location in enumerate(channelLocations, 0):
                        filename = str(i) + '.hdf5'
                        with h5py.File(writeDir/filename, 'a') as f:
                            f['stack'][time] = image[:,
                                        location - channelWidth : location + channelWidth]
                
                sys.stdout.write(f"{len(channelLocations)} from position: {position} and time: {time} written\n")
                sys.stdout.flush()
        elif imageType == 'barcodes':
            positionDir = str(position)
            barcodesDir = mainAnalysisDir / positionDir/ imageType

            if not barcodesDir.exists():
                barcodesDir.mkdir(parents=True, exist_ok=True)

            for i, oneBarcode in enumerate(image, 0):
                filename = str(time) + "_" + str(i) + '.tiff'
                oneBarcodeFilename = barcodesDir / filename
                io.imsave(oneBarcodeFilename, oneBarcode, check_contrast=False, compress=6, plugin='tifffile')
            
            sys.stdout.write(f"{len(image)} barcodes written to disk\n")
            sys.stdout.flush()


    # return the number of channels detected, locations to write to database 
    # assume it is one image per batch
    # TODO: batching of images done later
    def processChannels(self, image, position, time):
        
        # pass throught the cell net to get 
        sys.stdout.write(f"Image is on GPU: {image.is_cuda} -- \n")
        sys.stdout.flush()
        cellSegMask = torch.sigmoid(self.cellSegNet(image)) > self.cellProcessParameters['segThreshold']

        # send to cpu to be saved cut according to position
        cellSegMaskCpu = cellSegMask.cpu().detach().numpy().squeeze(0).squeeze(0)

        # remove smaller objects
        #cellSegMaskCpu = remove_small_objects(cellSegMaskCpu.astype('bool'), min_size=self.cellProcessParameters['smallObjectsArea'])

        self.writeFileH5Py(cellSegMaskCpu, 'cellSegmentation', position, time)

        # get the phase image and use it to crop channels for viewing
        phase_img = image.cpu().detach().numpy().squeeze(0).squeeze(0)


        self.writeFileH5Py(phase_img, 'phaseFullImage', position, time)
        # pass through net and get the results
        # change of approach, we only find channels in the first image and use them for the rest of the
        # images as it is possible to accumulate errors in wierd ways if you use channel locations from
        # each image, especially in 40x on data of not the highest quality
        if time == 0:
            channelSegMask = torch.sigmoid(self.channelSegNet(image)) > self.channelProcessParameters['segThreshold']

            # sent to cpu and saved according to position and timepoint
            channelSegMaskCpu = channelSegMask.cpu().detach().numpy().squeeze(0).squeeze(0)

            # need to remove smaller objects of the artifacts
            #channelSegMaskCpu = remove_small_objects(channelSegMaskCpu.astype('bool'), min_size = self.channelProcessParameters['smallObjectsArea'])
            self.writeFileH5Py(channelSegMaskCpu, 'channelSegmentation', position, time)
            
            hist = np.sum(channelSegMaskCpu, axis = 0) > self.channelProcessParameters['minChannelLength']

            peaks, _ = find_peaks(hist, distance=self.channelProcessParameters['minPeaksDistance'], 
                            plateau_size=self.channelProcessParameters['plateauSize'])
        
            locationsBarcodes, locationsChannels = findBarcodesAndChannels(peaks, 
                                    self.channelProcessParameters)


            # grab barcode and then grab the channels in each image and write
            barcodeImages = []
            barcodeWidth = self.channelProcessParameters['barcodeWidth']
            for location in locationsBarcodes:
                barcode_img = phase_img[:, location - barcodeWidth//2: location + barcodeWidth//2]
                barcodeImages.append(barcode_img)
            # stack the barcode and write all at the same time
            self.writeFileH5Py(barcodeImages, 'barcodes', position, time)
            sys.stdout.write(f"No of barcode regions detected: {len(barcodeImages)}\n")
            sys.stdout.flush()

            if len(locationsChannels) == 0:
                sys.stdout.write(f"Skipping position: {position} data\n")
                sys.stdout.flush()
                
                # record failed status to the database

            else:
                # write the channels appropraitely
                sys.stdout.write(f"No of channels identified: {len(locationsChannels)}\n")
                sys.stdout.flush()
                
                # write the phase and segmented mask chopped files
                #self.writeFileH5Py(phase_img, 'oneMMChannelPhase', position, time, channelLocations=locationsChannels)

                #self.writeFileH5Py(cellSegMaskCpu, 'oneMMChannelCellSeg', position, time, channelLocations=locationsChannels)

            # write positions to database
            dataToDatabase = {
                'time': time,
                'position': position,
                'locations': pickle.dumps(locationsChannels),
                'numchannels': len(locationsChannels)
            }
            self.recordInDatabase('segment', dataToDatabase)

        else:
            # what to do for the rest of the timepoint, use the positions from above
            # get channel locations from the database
            locationsChannels = self.getLocationsFromDatabase('segment', position, 0)

            # write phase images
            #self.writeFileH5Py(phase_img, 'oneMMChannelPhase', position, time, channelLocations=locationsChannels)

            # write cell segmentation images
            #self.writeFileH5Py(cellSegMaskCpu, 'oneMMChannelCellSeg', position, time, channelLocations=locationsChannels)
            
            dataToDatabase = {
                'time': time, 
                'position': position,
                'locations': pickle.dumps(locationsChannels),
                'numchannels': len(locationsChannels)
            }

            self.recordInDatabase('segment', dataToDatabase)
            
        sys.stdout.write("\n ---------\n")
        sys.stdout.flush()

        return locationsChannels


    def processCells(self, image, position, time, channelLocations):
        
        # pass throught the cell net to get 
        cellSegMask = torch.sigmoid(self.cellSegNet(image)) > self.cellProcessParameters['segThreshold']

        # send to cpu to be saved cut according to position
        cellSegMaskCpu = cellSegMask.cpu().detach().numpy().squeeze(0).squeeze(0)

        # remove smaller objects
        cellSegMaskCpu = remove_small_objects(cellSegMaskCpu.astype('bool'), min_size=self.cellProcessParameters['smallObjectsArea'])

        self.writeFile(cellSegMaskCpu, 'cellSegmentation', position, time)
    
    
    def segment(self):
        # segmentation loop for both cell and channels
        sys.stdout.write(f"Starting segmentation ... \n")
        sys.stdout.flush()
        self.loadNets()

        while not self.segmentKillEvent.is_set():
            try:
                dataloader = DataLoader(self.segmentDataset, batch_size=1)
                with torch.no_grad():
                    for data in dataloader:
                        #image = data['image'].to(self.device)
                        image = data['image'].to(self.device)
                        if data == None:
                            #time.sleep(2)
                            continue
                        channelLocations = self.processChannels(image, int(data['position']), int(data['time']))
                        # put the datapoint in the queue for calculating the growth stuff like areas, lengths, etc
                        del image

                        self.writeQueue.put({
                            'position': int(data['position']), 
                            'time': int(data['time']),
                            'numchannels': len(channelLocations)
                                    })

                        #sys.stdout.write(f"Image shape segmented: {image.shape}--{data['position']} -- {data['time']} \n")
                        #sys.stdout.flush()

            except Empty:
                sys.stdout.write("Segmentation queue is empty .. but process shutdown is not happening\n")
                sys.stdout.flush()
            except KeyboardInterrupt:
                self.segmentKillEvent.set()
                sys.stdout.write(f"Segmetation process interrupted using keyboard\n")
                sys.stdout.flush()

        sys.stdout.write("Segmentation process completed successfully\n")
        sys.stdout.flush()

    def calculateOnePosition(self, datapoint):
        # calculate the properties of one position and write them to the database
        try:
            
            mainAnalysisDir = Path(self.imageProcessParameters["saveDir"])
            position = int(datapoint[0])
            time = int(datapoint[1])

            channelLocations = self.getLocationsFromDatabase('segment', int(datapoint[0]), 0)
            # get channel locations from database
            filename = str(time) + '.tiff'
            positionDir = str(position)

            phaseImageFilename = mainAnalysisDir / positionDir / "phaseFullImage" / filename
            segImageFilename = mainAnalysisDir / positionDir / "cellSegmentation" / filename

            # read the phase image cut and write
            phase_img = io.imread(phaseImageFilename)
            seg_img = io.imread(segImageFilename) * 255
            seg_img = seg_img.astype('uint8')

            # data for one image is bundled and added to the database at once
            dataToDatabase = []

            channelWidth = self.channelProcessParameters['channelWidth'] // 2
            for (i, location) in enumerate(channelLocations, 0):
                channelNo = str(i)
                phaseChannelsDir  = mainAnalysisDir / positionDir/ "oneMMChannelPhase" / channelNo
                segChannelsDir = mainAnalysisDir / positionDir / "oneMMChannelCellSeg" / channelNo
                if not phaseChannelsDir.exists():
                    phaseChannelsDir.mkdir(parents=True, exist_ok=True)
                
                if not segChannelsDir.exists():
                    segChannelsDir.mkdir(parents=True, exist_ok=True)

                phaseChannelImg = phase_img[:,
                                    location - channelWidth: location + channelWidth]
                segChannelImg = seg_img[:,
                                    location - channelWidth: location + channelWidth]
                # write the image
                phaseChannelFileName = phaseChannelsDir / filename
                segChannelFileName = segChannelsDir / filename

                props = regionprops(label(segChannelImg))
                areas = []
                lengths = []
                numobjects = []
                for i in range(len(props)):
                    if props[i]['area'] > 64 and props[i]['major_axis_length'] < 200:
                        areas.append(props[i]['area'])
                        lengths.append(props[i]['major_axis_length'])
                        numobjects.append(i)

                channelPropertiesToDatabase = {
                    'position': position,
                    'timepoint': time,
                    'channelno': i, 
                    'areas': pickle.dumps(areas),
                    'lengths': pickle.dumps(lengths),
                    'numobjects': pickle.dumps(numobjects)
                } 
                
                io.imsave(phaseChannelFileName, phaseChannelImg.astype('float16'), check_contrast=False, compress = 6, plugin='tifffile')
                io.imsave(segChannelFileName, segChannelImg.astype('uint8'), check_contrast=False, compress=6, plugin='tifffile')

                dataToDatabase.append(channelPropertiesToDatabase)

            self.recordInDatabase('growth', dataToDatabase)

            sys.stdout.write(f"Calculating for position: {datapoint[0]} -- time: {datapoint[1]} -- no of channels: {len(channelLocations)}\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"Error : {e} in writing files\n")
            sys.stdout.flush()


        #self.recordInDatabase('growth', properties)

    def properties(self):
        # dead-alive net loop for doing dead-alive analysis in single channel phase stacks
        sys.stdout.write(f"Starting properties analyzer ... \n")
        sys.stdout.flush()

        # wait for kill event
        while not self.writeKillEvent.is_set():
            try:
                # write the dataloader to get the right stuff into the net
                dataloader = DataLoader(self.writeDataset, batch_size=6, num_workers=2)
                with torch.no_grad():
                    for data in dataloader:
                        #calculateOnePosition(data['position'], data['time'], data['numChannels'])
                        if data is None:
                            continue
                        else:
                            # arguments construction for pool execution
                            positions = list(data['position'].numpy())
                            times = list(data['time'].numpy())
                            numOfChannels = list(data['numchannels'].numpy())
                            arguments = list(zip(positions, times, numOfChannels))
                        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                            executor.map(self.calculateOnePosition, arguments)

                        # start a thread pool to speed up the execution of reading writing properties
                        sys.stdout.write(f" Write Process: {data}\n")
                        sys.stdout.flush()
                        
            except KeyboardInterrupt:
                self.writeKillEvent.set()
                sys.stdout.write("Writing process interrupted using keyboard\n")
                sys.stdout.flush()

        
        sys.stdout.write("Writing properties process completed successfully\n")
        sys.stdout.flush()

    # this function will be called for calculating the growth for a position
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
        self.acquireKillEvent.set()
        self.segmentKillEvent.set()
        self.writeKillEvent.set()
    # if it fails write the state and bail, and use this state to restart after adjusting 
    def savedState(self):
        pass
    
def runProcesses(exptRunObject):
    #exptRunObject.loadNets()
    try:
        mp.set_start_method('spawn')
    except:
        pass
    exptRunObject.acquireKillEvent.clear()
    acquireProcess = mp.Process(target=exptRunObject.acquireFake, name='Acquire Process')
    acquireProcess.start()

    exptRunObject.segmentKillEvent.clear()
    segmentProcess = mp.Process(target=exptRunObject.segment, name='Segment Process')
    segmentProcess.start()

    exptRunObject.writeKillEvent.clear()
    writeProcess = mp.Process(target=exptRunObject.properties, name='Propertis write Process')
    writeProcess.start()

# In the datasets image names are img_000000000.tiff format.
def imgFilenameFromNumber(number):
    if number == 0:
        num_digits = 1
    else:
        num_digits = int(math.log10(number)) + 1
    imgFilename = 'img_' + '0' * (9 - num_digits) + str(number) + '.tiff'
    return imgFilename

def findBarcodesAndChannels(peaks, parameters = { 'minChannelLength': 200, 'minPeaksDistance' : 25, 
                    'barcodeWidth' : 48, 'channelsPerBlock': 21, 'plateauSize':15, 'channelWidth': 36}):
    
    #hist = np.sum(image, axis = 0) > parameters['minChannelLength']

    #peaks, _ = find_peaks(hist, distance=parameters['minPeaksDistance'], plateau_size=parameters['plateauSize'])
    
    indices_with_larger_gaps = np.where(np.ediff1d(peaks) > parameters['barcodeWidth'])[0]
    
    locations_before_barcode = peaks[indices_with_larger_gaps]
    locations_after_barcode = peaks[indices_with_larger_gaps + 1]
    
    locations_barcode = np.rint(np.mean((locations_before_barcode,
                                        locations_after_barcode), axis = 0)).astype('int')
    
    num_barcodes = len(locations_barcode)
    # there are 5 barcodes seen in the image
    if num_barcodes == 5:
        # count the number of channels before the first barcode and after the 
        # last barcode and include them upto numChannels channels
        y_channels = []
        
        # channels before first barcode
        indices_before_first = np.where(peaks < locations_barcode[0])[0]
        if peaks[indices_before_first[0]] < parameters['channelWidth']//2:
            indices_before_first = indices_before_first[1:]

        y_channels.extend(list(peaks[indices_before_first]))
        
        for i in range(num_barcodes):
            indices = np.where(np.logical_and(peaks > locations_barcode[i-1],
                                             peaks < locations_barcode[i]))[0]
            y_channels.extend(list(peaks[indices]))
            
        # number of channels to count after the last
        number_to_include = parameters['channelsPerBlock'] - len(indices_before_first)
        indices_after_last = np.where(peaks > locations_barcode[-1])[0]
        y_channels.extend(list(peaks[indices_after_last][:number_to_include]))
        
    elif num_barcodes == 6:
        y_channels = []
        # count only the channels between barcodes and 
        # grab the (x, y) locations to cut,
        # x will be the top of the channel, row number
        # y will be the peak picked up in the histogram, between the barcodes
        # count 21 channels after calculating
        for i in range(num_barcodes):
            indices = np.where(np.logical_and(peaks > locations_barcode[i-1],
                                             peaks < locations_barcode[i]))[0]
            #if len(indices) == 21:
            # all good pick them up
            y_channels.extend(list(peaks[indices]))   
        
    else:
        # detection failure, since it is ambiguous skipp the position
        y_channels = []
        sys.stdout.write(f"Detection failure, {num_barcodes} detected\n")
        sys.stdout.flush()
    # locations of the barcode and locations of channels to cut.
    return locations_barcode, y_channels

class tweezerWindow(QMainWindow):

    def __init__(self):
        pass

if __name__ == "__main__":
    print("Experiment Processes launch ...")
    # parse the argments and create appropriate processes and queues