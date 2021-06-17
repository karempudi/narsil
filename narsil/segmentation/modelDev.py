import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from numpy.lib.npyio import save
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage import transform
from skimage import measure, img_as_ubyte
from skimage.morphology import binary_dilation
from skimage.io import imread, imsave
from narsil.utils.transforms import  resizeOneImage, tensorizeOneImage
from narsil.utils.losses import WeightedUnetLoss, UnetLoss, WeightedUnetLossExact
from .datasets import mmDataMultiSpecies, phaseFolder, phaseTestDir
from .network import basicUnet, smallerUnet 

class trainNet(object):

	def __init__(self, dataDir, species, transforms, modelParameters, optimizationParameters):
		self.dataDir = dataDir
		self.species = species
		self.transforms = transforms

		self.modelParameters = modelParameters
		self.optimizationParameters = optimizationParameters

		
		# device initialization
		self.device = torch.device(self.modelParameters['device'] if torch.cuda.is_available() else "cpu")

		# initalize dataset
		self.initializeDataset()

		# initialize net
		self.initializeNet()

		# initialize optimizer
		self.initializeOptimizer()

		# initialize loss function
		self.initializeLossFunction()

		self.sendToDevice()


	def initializeDataset(self):
		self.trainDataset = mmDataMultiSpecies(self.dataDir, self.species, self.transforms, datasetType='train',
									includeWeights=self.modelParameters['includeWeights']) 
		# on validationDataset, validate on the full image, instead of the crops
		self.validationDataset = mmDataMultiSpecies(self.dataDir, self.species, transforms=None, datasetType='validation')

		# dataLoaders initialization
		self.trainDataLoader = DataLoader(self.trainDataset, batch_size = self.optimizationParameters['batchSize'],
									shuffle=True, num_workers = self.optimizationParameters['cores'])
		print("Training DataLoaders initialized .... ")
		self.validationDataLoader = DataLoader(self.validationDataset, batch_size=1, 
									shuffle=False, num_workers=self.optimizationParameters['cores'])
		print("Validation DataLoaders initialized .... ")

		

	def initializeNet(self):
		if self.modelParameters['netType'] == 'big':
			self.net = basicUnet(self.modelParameters['transposeConv'])
		elif self.modelParameters['netType'] == 'small':
			self.net = smallerUnet(self.modelParameters['transposeConv'])
		
		print("Network initalized successfully ...")

	def initializeLossFunction(self):
		#self.lossFunction = nn.BCELoss()
		#self.lossFunction = WeightedUnetLoss()
		self.lossFunction = WeightedUnetLossExact()

		print("Loss function initialized successfully ...")

	def initializeOptimizer(self):
		self.optimizer = optim.Adam(self.net.parameters(),lr = self.optimizationParameters['learningRate'])
		self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
							step_size = self.optimizationParameters['schedulerStep'],
							gamma = self.optimizationParameters['schedulerGamma'] )
		
		print("Optimizer and scheduler initialized successfully ... ")


	def sendToDevice(self):
		self.net.to(self.device)

	def train(self):
		print("Training started .... ")
		self.losses = []
		for epoch in range(self.optimizationParameters['nEpochs']):
			epochAvgLoss = self.runEpoch(epoch)
			print(f"Epoch {epoch} done --- {epochAvgLoss}")
			self.losses.append(epochAvgLoss)
			self.scheduler.step()

	def runEpoch(self, epoch):
		epochLoss = 0.0
		for i_batch, data in enumerate(self.trainDataLoader, 0):

			phase, mask, weights= data['phase'].to(self.device), data['mask'].to(self.device), data['weights'].to(self.device)

			self.optimizer.zero_grad()

			mask_pred = self.net(phase)
			loss = self.lossFunction(mask_pred, mask, weights)

			epochLoss += loss.item()
			print(f"Epoch : {epoch + 1}, ImageBatch: {i_batch} -- loss: {loss.item()}")

			loss.backward()
			self.optimizer.step()
		
		# run the model through a validation loop
		return epochLoss/i_batch
		

	def plotLosses(self):
		pass

	def plotData(self, idx):

		fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
		datapoint = self.trainDataset[idx]
		ax1.imshow(datapoint['phase'].numpy().squeeze())
		ax2.imshow(datapoint['mask'].numpy().squeeze())
		ax3.imshow(datapoint['weights'].numpy().squeeze())
		ax1.set_xlabel(datapoint['phaseFilename'])
		ax2.set_xlabel(datapoint['maskFilename'])
		ax3.set_xlabel(datapoint['weightsFilename'])
		plt.show()


	def save(self, path):
		transformsUsed = str(self.transforms)
		
		savedModel = {
			'modelParameters' : self.modelParameters,
			'optimizationParameters': self.optimizationParameters,
			'trasnformsUsed' : transformsUsed,
			'species' : self.species,
			'model_state_dict': self.net.state_dict(),
		}
		torch.save(savedModel,path)



def testNet(modelPath, phaseDir, saveDir, transforms,
	threshold = None, device = 'cuda:1', fileformat ='.tiff', 
	plotContours='False', contoursThreshold=0.9):
	
	savedModel = torch.load(modelPath)
	if savedModel['modelParameters']['netType'] == 'big':
		net = basicUnet(savedModel['modelParameters']['transposeConv'])
	elif savedModel['modelParameters']['netType'] == 'small':
		net = smallerUnet(savedModel['modelParameters']['transposeConv'])

	device = torch.device(device if torch.cuda.is_available() else "cpu")

	#resizing = resizeOneImage((1040, 2048), (1040, 2048))
	#resizing = resizeOneImage((2048, 4096), (1024, 2048))
	#tensorizing = tensorizeOneImage(1)

	#transform = transforms.Compose([resizing, tensorizing])

	motherMachineDataset = phaseTestDir(phaseDir, transform=transforms, phase_fileformat=fileformat,
				addNoise=False, flip=False)

	motherMachineDataLoader = DataLoader(motherMachineDataset, batch_size=1, shuffle=True, num_workers=6)

	net.load_state_dict(savedModel['model_state_dict'])
	net.to(device)
	net.eval()

	numTest = len(motherMachineDataset)
	print(f"Length of dataset is {numTest}")

	# test loop
	with torch.no_grad():
		for i_batch, data in enumerate(motherMachineDataLoader, 0):

			# get the image tensors
			phase = data.to(device)
			# zero the paramter gradients
			# forward + backward + optimize
			mask_pred = net(phase) 

			# set sigmoid if the net gives 
			mask_pred = torch.sigmoid(mask_pred)

			#mask_pred = torch.sigmoid(mask_pred)
			if threshold != None:
				mask_pred = mask_pred.to("cpu").numpy().squeeze(0).squeeze(0) >= threshold
			else:
				mask_pred = mask_pred.to("cpu").numpy().squeeze(0).squeeze(0) 

			phase_save = phase.to("cpu").numpy().squeeze(0).squeeze(0)

			if plotContours == False:
				plt.figure()
				plt.imshow(mask_pred)
				plt.show(block=False)

			if plotContours == True:
				contours = measure.find_contours(mask_pred, contoursThreshold)
				fig, ax = plt.subplots()
				ax.imshow(phase_save, cmap= plt.cm.gray)

				for contour in contours:
					ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')
				plt.show(block=False)
#        print(f"ImageBatch: {i_batch} -- loss: {loss.item()}")
			if saveDir != None:
				imsave(saveDir + str(i_batch) + '.tif', img_as_ubyte(mask_pred), plugin='tifffile', compress = 6, check_contrast=False)
	#    print(f"Epoch {epoch+1} Done --- Loss per Image: {epoch_loss/numTrain}")


			if i_batch == 10:
				break

			

	

	


