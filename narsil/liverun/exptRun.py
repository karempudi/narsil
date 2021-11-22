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
        self.deadaliveQueue = mp.Queue()

        #self.acquireProcess = None
        #self.segmentProcess = None
        #self.deadAliveProcess = None

        self.acquireKillEvent = mp.Event()
        self.segmentKillEvent = mp.Event()
        self.deadaliveKilEvent = mp.Event()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # datasets: These are wrappers around torch multiprocessing queues, that are used
        # to fetch data using iterable dataloader. Dataloader
        self.segmentDataset = queueDataset(self.segmentQueue) 


        self.cellSegNet = None
        self.channelSegNet = None


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

        except pgdatabase.DatabaseError as e:
            sys.stderr.write(f"Error in writing to database: {e}\n")
            sys.stderr.flush()
        finally:
            if con:
                con.close()
    
    def getFromDatabase(self, tableName, position, time):
        con = None
        try:
            con = pgdatabase.connect(database=self.dbParameters['dbname'],
                                    user=self.dbParameters['dbuser'],
                                    password=self.dbParameters['dbpassword'])
            cur = con.cursor()
            con.autocommit = True

            if tableName == 'segment':
                cur.exectue("SELECT locations FROM segment WHERE position=%s AND time=%s", (position, time))

                # you get a pickled bytear that needs to be converted to numpy
                rows = cur.fetchall()

                channelLocations = pickle.loads(rows[0])

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

            time.sleep(0.5)

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

            sys.stdout.write(f"{cellMaskFilename} written \n")
            sys.stdout.flush()

            image  = image * 255
            io.imsave(cellMaskFilename, image.astype('uint8'), compress=6, check_contrast=False,
                        plugin='tifffile')
            sys.stdout.write(str(cellMaskFilename) + " written \n")
            sys.stdout.flush()
            
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
                    io.imsave(channelFileName, channelImg, check_contrast=False, compress=6, plugin='tifffile')
            
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
        cellSegMaskCpu = remove_small_objects(cellSegMaskCpu.astype('bool'), min_size=self.cellProcessParameters['smallObjectsArea'])

        self.writeFile(cellSegMaskCpu, 'cellSegmentation', position, time)

        # get the phase image and use it to crop channels for viewing
        phase_img = image.cpu().detach().numpy().squeeze(0).squeeze(0)
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
            self.writeFile(channelSegMaskCpu, 'channelSegmentation', position, time)

            locationsBarcodes, locationsChannels = findBarcodesAndChannels(channelSegMaskCpu, 
                                    self.channelProcessParameters)


            # grab barcode and then grab the channels in each image and write
            barcodeImages = []
            barcodeWidth = self.channelProcessParameters['barcodeWidth']
            for location in locationsBarcodes:
                barcode_img = phase_img[:, location - barcodeWidth//2: location + barcodeWidth//2]
                barcodeImages.append(barcode_img)
            # stack the barcode and write all at the same time
            self.writeFile(barcodeImages, 'barcodes', position, time)
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
                self.writeFile(phase_img, 'oneMMChannelPhase', position, time, channelLocations=locationsChannels)

                self.writeFile(cellSegMaskCpu, 'oneMMChannelCellSeg', position, time, channelLocations=locationsChannels)

            # write positions to database
            dataToDatabase = {
                'time': time,
                'position': position,
                'locations': pickle.dumps(locationsChannels),
                'numChannels': len(locationsChannels)
            }
            self.recordInDatabase('segment', dataToDatabase)

        else:
            # what to do for the rest of the timepoint, use the positions from above
            # get channel locations from the database
            locationsChannels = self.getFromDatabase('segment', position, 0)

            # write phase images
            self.writeFile(phase_img, 'oneMMChannelPhase', position, time, channelLocations=locationsChannels)

            # write cell segmentation images
            self.writeFile(cellSegMaskCpu, 'oneChannelCellSeg', position, time, channelLocations=locationsChannels)
            
            dataToDatabase = {
                'time': time, 
                'position': position,
                'locations': pickle.dumps(locationsChannels),
                'numChannels': len(locationsChannels)
            }

            self.recordInDatabase('segment', dataToDatabase)
            
        sys.stdout.write("\n ---------\n")
        sys.stdout.flush()

        return None


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
                            time.sleep(2)
                            continue
                        self.processChannels(image, int(data['position']), int(data['time']))
                        del image

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

    def processDeadAlive(self, data):

        # pass the data through the net
        # and then write the hidden state to the database
        pass


    def deadalive(self):
        # dead-alive net loop for doing dead-alive analysis in single channel phase stacks
        sys.stdout.write(f"Starting dead-alive analyzer ... \n")
        sys.stdout.flush()

        # wait for kill event
        while not self.deadaliveKillEvent.is_set():
            try:
                time.sleep(2)

                # write the dataloader to get the right stuff into the net
                dataloader = DataLoader(self.deadAliveDataset, batch_size=20, num_workers=1)
                with torch.no_grad():
                    for data in dataloader:
                        data = data
                        self.processDeadAlive(data,)
                        
            except KeyboardInterrupt:
                self.deadaliveKillEvent.set()
                sys.stdout.write("Dead alive process interrupted using keyboard\n")
                sys.stdout.flush()

        
        sys.stdout.write("Dead Alive process completed successfully\n")
        sys.stdout.flush()

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

    #exptRunObject.deadaliveKillEvent.clear()
    #deadAliveProcess = tmp.Process(target=exptRunObject.deadalive, name='DeadAlive Process')
    #deadAliveProcess.start()

# In the datasets image names are img_000000000.tiff format.
def imgFilenameFromNumber(number):
    if number == 0:
        num_digits = 1
    else:
        num_digits = int(math.log10(number)) + 1
    imgFilename = 'img_' + '0' * (9 - num_digits) + str(number) + '.tiff'
    return imgFilename

def findBarcodesAndChannels(image, parameters = { 'minChannelLength': 200, 'minPeaksDistance' : 25, 
                    'barcodeWidth' : 48, 'channelsPerBlock': 21, 'plateauSize':15}):
    
    hist = np.sum(image, axis = 0) > parameters['minChannelLength']

    peaks, _ = find_peaks(hist, distance=parameters['minPeaksDistance'], plateau_size=parameters['plateauSize'])
    
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