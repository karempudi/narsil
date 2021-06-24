# Training tracker neural nets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from narsil.utils.losses import ContrastiveLoss

from torch.utils.data import Dataset, DataLoader
from narsil.tracking.siamese.network import siameseNet
from narsil.tracking.siamese.trainDatasets import siameseDatasetWrapper

class trainNet(object):

    def __init__(self, trainingDirs, transforms, modelParameters, optimizationParameters, 
                    validation=True):
        self.trainingDirs = trainingDirs
        self.transforms = transforms
        self.modelParameters = modelParameters
        self.optimizationParameters = optimizationParameters
        self.validation = validation
    	
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
        # setting dataloaders
        self.trainingDataset = []
        self.validationDataset = []

        self.trainDataLoader = DataLoader()
        self.validationDataLoader = DataLoader()

    def initializeNet(self):
        self.net = siameseNet(outputFeatureSize=self.modelParameters['outputFeatureSize'])

    def initializeOptimizer(self):
        self.optimizer = optim.Adam(self.net.parameters(), lr = self.optimizationParameters['learningRate'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                            step_size = self.optimizationParameters['schedulerStep'],
                            gamma = self.optimizationParameters['schedulerGamma'])

        print("Optimizer and scheduler initialized successfully ... ")

    def initializeLossFunction(self):
        self.lossFunction = ContrastiveLoss(margin = self.modelParameters['margin'])

        print("Loss function initialized successfully ... ")
    

    def train(self):
        print("Training started .... ")
        self.losses_train = []
        self.losses_validation = []
        for epoch in range(self.optimizationParameters['nEpochs']):
            epoch_train_loss = self.trainEpoch(epoch)
            print(f"Training Epoch {epoch + 1} done --- {epoch_train_loss}")
            self.losses_train.append(epoch_train_loss)
        
            # validation
            if self.validation == True:
                epoch_validation_loss = self.validateEpoch(epoch)
                self.losses_validation.append(epoch_validation_loss)
            
            self.scheduler.step()

    def trainEpoch(self, epoch):
        epochLoss = 0.0
        for i_batch, data in enumerate(self.trainDataLoader, 0):

            props1, image1, props2, image2, label = data[0]['props'].to(self.device), data[0]['image'].to(self.device), data[1]['props'].to(self.device), data[1]['props'].to(self.device), data[2].to(self.device)
            self.optimizer.zero_grad()
            output1, output2 = self.net(props1, image1, props2, image2)
            loss = self.lossFunction(output1, output2) 

            epochLoss += loss.item()
            print(f"Epoch : {epoch + 1}, Bundle Batch: {i_batch} -- loss: {loss.item()}")

            loss.backward()
            self.optimizer.step()
        
        return epochLoss/i_batch

    def validateEpoch(self, epoch):
        self.net.eval()
        validation_epoch_loss = 0.0
        with torch.no_grad():
            for i_batch, data in enumerate(self.validationDataLoader, 0):

                props1, image1, props2, image2, label = data[0]['props'].to(self.device), data[0]['image'].to(self.device), data[1]['props'].to(self.device), data[1]['props'].to(self.device), data[2].to(self.device)
                output1, output2 = self.net(props1, image1, props2, image2)
                loss_per_batch = self.lossFunction(output1, output2)
                
                validation_epoch_loss += loss_per_batch.item()
        # reset net to train mode after evaluating
        self.net.train()
        print(f"Validation run of epoch {epoch + 1} done ... ")
        return validation_epoch_loss/i_batch

    def sendToDevice(self):
        self.net.to(self.device)

    def plotLosses(self, ylim=[0, 2.0]):
        pass

    def plotData(self, idx):
        pass

    def save(self, path):

        transformsUsed = str(self.transforms)
        savedModel = {
			'modelParameters' : self.modelParameters,
			'optimizationParameters': self.optimizationParameters,
			'trasnformsUsed' : transformsUsed,
			'model_state_dict': self.net.state_dict(),
		}
        torch.save(savedModel,path)

