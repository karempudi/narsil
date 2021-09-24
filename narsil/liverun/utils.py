# Utilites like locked arrays and datastructrues/classes that handle stuff outside
from torch.utils.data import DataLoader, Dataset, IterableDataset
import multiprocessing
import tkinter as tk
from tkinter import filedialog
import sys
import json
import torch
from collections import OrderedDict
from skimage import transform

class resizeOneImage(object):

	def __init__(self, imgReizeShape, imgToNetSize):
		assert isinstance(imgReizeShape, tuple)
		assert isinstance(imgToNetSize, tuple)
		self.imgResizeShape = imgRezieShape
		self.imgToNetSize = imgToNetSize
	
	def __call__(self, image):
		height, width = image.shape

		image = np.pad(image, pad_width=((0, self.imgResizeShape[0] - height), (0, self.imgResizeShape[1] - width)),
					mode='constant', constant_values = 0.0)
		
		if self.imgResizeShape[0] != self.imgToNetSize[0] or self.imgResizeShape[1] != self.imgToNetSize[1]:
			image = transform.resize(image, self.imgToNetSize, anti_aliasing=True) 
		
		return image

class tensorizeOneImage(object):
	
	def __call__(self, phaseImage):
		phaseImage = phaseImage.astype('float32')
		return torch.from_numpy(phaseImage).unsqueeze(0)


"""
Gerenric queue for yielding objects from the queue in a safe way to be
processed by different processes piping into the queue and out of the
queue
"""
class queueDataset(IterableDataset):
	def __init__(self, queue):
		self.queue = queue
	
	def getNextImage(self):
		#print(f"Queue size in datalaoder: {self.queue.qsize()}")
		while self.queue.qsize() > 0:
			yield self.queue.get()
		return None

	def __iter__(self):
		return self.getNextImage()
	

class RNNQueue(IterableDataset):
	
	def __init__(self, queue, channelsWritePath):
		self.queue = queue
		self.lock = multiprocessing.Lock()
		self.channelsWritePath = channelsWritePath

	def getNextItem(self):
		# use locks and get the LSTM states that are written down in 
		# appropriate folders
		self.lock.acquire()
		while self.queue.qsize() > 0:
			channel = self.queue.get()
			# go get the lstm stacks, the release the locks


			self.lock.release()
			yield {'lstm': None, 'position': None, 'time': None}

		return None

	def __iter__(self):
		return self.getNextItem()
	
def lockedNumpyArray():
	pass

"""
Use filedialog on the GUI to click and open the correction postions list 
file that is saved using save Positions in Stage Position List in 
Micromanager 2.0
"""
def getPositionFileName():
	try:
		while True:
			root = tk.Tk()
			root.withdraw()
			positionFilePath = filedialog.askopenfilename()
			print(f"Position file path is: {positionFilePath}")
			return positionFilePath

	except KeyboardInterrupt:
		print("Keyboard Interrupted during position file setting")
		sys.exit()

""" 
Parse positions list file from micromanager 2.0 and return and orderedDict
of positions, PFS offset to be piped into creation of events for Pycromanager
"""
def parsePositionsFile(positionFileName, pfsDeviceIndex=0, XYStageIndex=1):
	print(f"Positions File name given is: \n")
	print(positionFileName)
	print("---------------------------------")
	# some constants for eading the data arrays from the positions file
	# this things might change names depending on the version of micromanger
	# so made them into variable, also double quotes are annoying to write
	mapping = "map"
	stagePositions = "StagePositions"
	array = "array"
	label = "Label"
	devicePositions = "DevicePositions"
	position_um = "Position_um"
	pfsDevice = pfsDeviceIndex
	xyDevice = XYStageIndex
	x = 0
	y = 1
	scalar = "scalar"
	positionsData = {}

	with open(positionFileName) as json_file:
		data = json.load(json_file)
		
		for item in data[mapping][stagePositions][array]:
			positionsData[item[label][scalar]] = {'x_coordinate': item[devicePositions][array][xyDevice][position_um][array][x],
				 'y_coordinate': item[devicePositions][array][xyDevice][position_um][array][y],
				 'pfs_offset': item[devicePositions][array][pfsDevice][position_um][array][x]
			}

	sorted_data = dict(sorted(positionsData.items(), key=lambda kv: int(kv[0][3:])))

	return OrderedDict(sorted_data)

"""
Function to create events that are used by pycromanager, 
things can change depending on how the presets loading happens 
internally on your microscope
"""
def phaseTimeSeriesEvents(positionsData, acquisitionParameters):
	events = []
	lastPosition = len(positionsData) - 1
	nTimePoints = acquisitionParameters['nTimePoints']
	timeInterval = acquisitionParameters['timeInterval']
	channelGroup = acquisitionParameters['channelGroup']
	channels = acquisitionParameters['channels']
	exposureTime_ms = acquisitionParameters['exposureTime_ms']
	for timePoint in range(nTimePoints):
		for i, position in enumerate(positionsData, 0):
			event = {}
			event['axes'] = {'time': timePoint, 'position': i}
			event['min_start_time'] = timePoint * timeInterval
			event['x'] = positionsData[position]['x_coordinate']
			event['y'] = positionsData[position]['y_coordinate']
			event['z'] = positionsData[position]['pfs_offset']

			if i == lastPosition:
				event['channel'] = {'group': channelGroup, 'config': channels[0]}
			else:
				event['channel'] = {'group': channelGroup, 'config': channels[1]}
			
			event['exposure'] = exposureTime_ms
			events.append(event)

	return events 

def getPositionList(filename=None):
    # if the filename is None, then just load something for test purposes
    if filename is None:
        positions = {}
        for i in range(20):
            positions['Pos' + str(i)] = {'x_coordinate':0, 'y_coordinate': 0, 'pfs_offset': 0}
        return positions
    else:
        positions = parsePositionsFile(filename)
        return positions


def getPositionsMicroManager():
    # connect to micromanager and get positions
    return None

def getMicroManagerPresets():
    return ["phase", "alexa488", "cy5", "cy3", "texasred"]
    

