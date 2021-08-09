# File to add all the functions that run in individual processes
from typing import Sequence
from pycromanager import Acquisition
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import torch.multiprocessing as tmp
from torch.utils.data import DataLoader
from utils import genericQueue, RNNQueue
from functools import partial
from queue import Empty
from utils import lockedNumpyArray
import torch.multiprocessing as tmp
import multiprocessing as mp
from narsil.liverun.utils import getPositionFileName, parsePositionsFile, phaseTimeSeriesEvents
import time
from torchvision import transforms
import psycopg2 as pg


def writeToDatabase(databaseParameters, tableName, data):
	# create a database connection
	dbname = databaseParameters['dbname']
	user = databaseParameters['dbuser']
	password = databaseParameters['dbpassword']

    if tableName == 'arrival':
		pass

	elif tableName == 'segmented':
    	pass

	elif tableName == 'deadAlive':
    	pass


"""
Function that will pipe stuff into the Segmentation queue. 
"""
def pipeToSegQueue(image, metadata, segQueue, imgArrivalQueue, imageSizeParameters):
	print("Image shape: ", image.shape)

	# put suff in the arrival queue to nofity the plotter
	# TODO: upgrade this to some form of database, with concurrent access 
	# for some realtime plotting instead of matplotlib plotting
	imgArrivalQueue.put({
		'time': metadata['Axes']['time'],
		'position': metadata['Axes']['position']
	})
	# Do some transforms to compose the image
	#******************************************

	# fill in the transformations later

	#******************************************
	#TODO:
	transforming = transforms.Compose([])
	try:
		# put the image and the metadata in the queue for segmentation
		i = int(time.time())
		imageTensor = transforming(image)
		#print(imageTensor.shape)
		print(f"----> {metadata['Axes']} ---- {metadata['Time']} -- {imageTensor.shape}")
		#print(f"---> {metadata['Time']}")
		segQueue.put({'number': i, 'image': imageTensor, 
					  'time': metadata['Axes']['time'],
					  'position': metadata['Axes']['position']})
		#print("Just put one image in the segmentation queue")
		#print(f"Queue size just after put: {segQueue.qsize()}")
	
	except Exception as error:
		print(f"Error in pipeToSegQueue -- image tensor not passed to segQueue --> {error}")



def acquisition(segQueue, imgArrivalQueue, acqShutDownEvent, acquisitionParameters):
	positionFilePath = getPositionFileName()
	positionData = parsePositionsFile(positionFilePath)
	for key, value in positionData.items():
		print(f"{key} -- X: {value['x_coordinate']} Y: {value['y_coordinate']}, PFS: {value['pfs_offset']}")

	# create events
	events = phaseTimeSeriesEvents(positionData, acquisitionParameters)

	print("Printing events .... ")
	for event in events:
		print('-------\n')
		print(event)

	# dummy is there as a hack to be able to send in extra arguments to 
	# pycromanager image_proces_fun hook. Generally it doesn't allow arguments
	# you will need to edit the source code to make this function happen..
	# Exact code will be forked and bundled later
	# The funciton hook take 2 or 4 arugments and we need more functionality
	# dummy is set to give 5 arguments and bypass the arugments check in pycromanager
	# arugmnet check for function hooks
	imageSizeParameters = acquisitionParameters['imageSizeParameters']
	with Acquisition(image_process_fn=partial(pipeToSegQueue, segQueue=segQueue, imgArrivalQueue=imgArrivalQueue,
	 				 imageSizeParameters=imageSizeParameters), debug=False) as acq:
		acq.acquire(events)
		print("Acquisition sent to micromanager ---- ")

	while not acqShutDownEvent.is_set():
		try:
			# print("Acquire is still running ...")
			time.sleep(2)
		except KeyboardInterrupt:
			acqShutDownEvent.set()
			print("Acquisition process interrupted using keyboard ... ")


def plotter():
	pass

def segCellsAndChannels():
	pass

def deadAlive():
	pass

def putDummyImages():
	pass