# Functions that can bundle channels from different positions 
import numpy as np
import glob
import os
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
import shutil
from torch.utils.data import Dataset, DataLoader
from narsil.fish.datasets import singleChannelFishData
from narsil.tracking.siamese.network import siameseNet
from narsil.tracking.siamese.datasets import singleChannelTrackingSiamese
from functools import partial
from multiprocessing import Pool

def evalifyNet(modelPath, device):
	device = torch.device(device if torch.cuda.is_available() else "cpu")
	
	savedModel = torch.load(modelPath)
	net = siameseNet(outputFeatureSize=savedModel['modelParameters']['outputFeatureSize'])
	net.share_memory()
	net.load_state_dict(savedModel['model_state_dict'])
	net.to(device)
	net.eval()
	for p in net.parameters():
		p.requires_grad_(False)
	
	return net

# Main function that processes tracks from one channel and calculated growth rates and species ID
def processOneChannel(channelDirName, net, bgFishData, trackingParameters):

	device = torch.load(trackingParameters['device'] if torch.cuda.is_available() else "cpu")
	batch_size = trackingParameters['batch_size']
	fluorChThreshold = trackingParameters['fluorChThreshold']
	channelSize = trackingParameters['channelSize']

	oneChannel = singleChannelTrackingSiamese(channelDirName, trackingParameters=trackingParameters,
				backgroundFishData=bgFishData, fluorChThreshold=fluorChThreshold, channelSize=channelSize)
	#print(len(oneChannel))
	# num_workers = 0 as the number of copies of the net are running at same time at process level
	# don't change it 
	stackdataloader = DataLoader(oneChannel, batch_size=batch_size, shuffle=False, num_workers=0)

	# results to keep track of scores
	prevFileNames = []
	currFileNames = []
	prevBlobNumbers = []
	currBlobNumbers = []
	scores = []

	# Run through the net and accumulte the results
	with torch.no_grad():
		for i, data in enumerate(stackdataloader, 0):
			blob1_props, blob1_image, blob2_props, blob2_image = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
			out1, out2 = net(blob1_props, blob1_image, blob2_props, blob2_image)
			net_scores = F.pairwise_distance(out1, out2, keepdim=True)
			prevFileNames.extend(data[4])
			prevBlobNumbers.extend(data[5].numpy())
			currFileNames.extend(data[6])
			currBlobNumbers.extend(data[7].numpy())
			output_scores = net_scores.to("cpu").detach().numpy()
			scores.extend(output_scores)
	# return prevFileNames, currFileNames, prevBlobNumbers, currBlobNumbers, scores	

	# Data goes into tracking above only if there are sufficent no of fluorescent channels lighting up 
	# you can always call oneChannel.createDataForNet() to let data for tracking be created
	if oneChannel.doneTracking:
		oneChannel.constructLinks(prevFileNames, currFileNames, prevBlobNumbers, currBlobNumbers, scores)
		oneChannel.convertLinksToTracks()
		oneChannel.labelTracksWithFluorChannels(printFluorLabels=False)
		oneChannel.setSpeciesForAllTracks()

		localGrowthRates = oneChannel.getGrowthOfSpeciesLabeledTracks()
		if trackingParameters['plot'] == True:
			#oneChannel.plotAllTracksWithFISH()
			oneChannel.plotTracksUsedForGrowth()

		if trackingParameters['writeGrowthRates'] == True:
			writeDirectory = channelDirName.split('blobs')[0] + str('growthRates/')
			channelNumber = channelDirName.split('blobs')[1].strip('/') # still a string
			if not os.path.exists(writeDirectory):
				os.makedirs(writeDirectory)
			
			writeFilename = writeDirectory + channelNumber + ".pickle"

			with open(writeFilename, "wb") as handle:
				pickle.dump(localGrowthRates, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	del oneChannel

def processOnePosition(positionDirName, net, trackingParameters):
	
	positionNumber = int(positionDirName.split('blobs')[0].split('/')[-2][3:])

	# we use background channel to be able to subtract background from the fish data
	if trackingParameters['backgroundChannelNumber'] != None and (positionNumber not in trackingParameters['flipPositions']):
		# set background fish directory
		backgroundFishDir = positionDirName.split('blobs')[0] + 'fishChannels/' + str(trackingParameters['backgroundChannelNumber']) + '/'
		bgFishData = singleChannelFishData(backgroundFishDir, channelNames=trackingParameters['channelNames'], transforms=None)
	elif trackingParameters['backgroundChannelNumber'] != None and (positionNumber in trackingParameters['flipPositions']):
		# set background fish directory
		backgroundFishDir = positionDirName.split('blobs')[0] + 'fishChannels/' + str(trackingParameters['backgroundChannelNumber'] + 1) + '/'
		bgFishData = singleChannelFishData(backgroundFishDir, channelNames=trackingParameters['channelNames'], transforms=None)
	else:
		bgFishData = None

	for i in range(trackingParameters['numChannels']):
		channeldirName = positionDirName + str(i) + '/'
		processOneChannel(channeldirName, net, bgFishData, trackingParameters=trackingParameters)

	print(f"{positionDirName.split('blobs')[0]} -- Done")
	

def processAllPositions(analysisDir, positions, trackingParameters, 
						skipPositions, numProcess=6):
	start = time.time()
	listPositionDirs = []
	for position in positions:
		if position not in skipPositions:
			listPositionDirs.append(analysisDir + 'Pos' + str(position) + '/blobs/')

	print(listPositionDirs)
	trackingnet = evalifyNet(trackingParameters['modelPath'], trackingParameters['device'])

	try:
		mp.set_start_method('spawn')
	except RuntimeError:
		pass

	pool = mp.Pool(processes=numProcess)
	pool.map(partial(processOnePosition, net=trackingnet, trackingParameters=trackingParameters), listPositionDirs)
	pool.close()
	pool.join()

	duration = time.time() - start

	print(f"Duration of tracking {len(positions)} positions is {duration}s")

def deleteTrackingData(analysisDir, positions, dirNameToDelete):

	dirsToDelete = []
	for position in positions:
		directory = analysisDir + 'Pos' + str(position) + '/' + dirNameToDelete + '/'
	
	for directory in dirsToDelete:
		try:
			shutil.rmtree(directory)
		except:
			print(f"Eror when deleting {directory} -- probably doesn't exist")
