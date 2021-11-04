import torch
import numpy as np
from torch.utils.data import DataLoader
from narsil.deadalive.network import deadAliveNet, deadAliveNet80036
from narsil.deadalive.datasets import channelStackTrain

class trainDeadAliveNet():
    def __init__(self, phaseDirectoriesList, modelParameters, optimizationParameters,
                    validation=True, validataionDirsList=None, fileformat='.tiff'):

        self.phaseDirectoriesList = phaseDirectoriesList
        self.modelParameters = modelParameters
        self.optimizationParameters = optimizationParameters
        self.device = torch.device(self.modelParameters['device'] if torch.cuda.is_available() else "cpu")
        self.validation = validation
        self.validationDirsList = validataionDirsList
        self.fileformat = fileformat
        self.initializeNet()

    def initializeNet(self):
        self.net = deadAliveNet80036(device=self.device)
        self.net.to(self.device)
        self.net.setup_optimizer(self.optimizationParameters['learning_rate'])

    # TODO: insert validation loader into the loop. 
    def train(self):
        numUnrolls = 2
        nEpochs = self.optimizationParameters['nEpochs']

        for epoch in range(nEpochs):
            stack = channelStackTrain(self.phaseDirectoriesList, numUnrolls, fileformat=self.fileformat)
            dataloader = DataLoader(stack, batch_size=16, shuffle=True, num_workers=6)

            epoch_loss = []
            print(f"Epoch {epoch} -- started")

            # get miniBatches and then in each minibatch loop over numUnrolls
            for i_batch, data in enumerate(dataloader, 0):
                lstm_state = None
                imageSequences, stateSequences = data['imageSequence'].to(self.device), data['statesSequence'].to(self.device)
                sequenceLosses = []
                sequenceTargets = []
                nTimeSteps = imageSequences.shape[-1]
                for timeStep in range(nTimeSteps):
                    self.net.optimizer.zero_grad()
                    output = self.net(imageSequences[:, :, :, :, timeStep], lstm_state)

                    sequenceLosses.append(output)
                    sequenceTargets.append(stateSequences[:, :, timeStep])
                    lstm_state = self.net.lstm_state
                loss = self.net.loss(torch.stack(sequenceLosses), torch.stack(sequenceTargets))
                loss.backward()
                self.net.optimizer.step()

                epoch_loss.append(loss.item())
            #lstm_state = self.net.lstm_state
            if epoch%10== 0:
                numUnrolls += 2

            print(f"Epoch average loss: {np.mean(epoch_loss)}")
            # run the validation loop

    def plotData(self, idx):
        pass

    def save(self, path):
        savedModel = {
            'modelParameters': self.modelParameters,
            'optimizationParamters': self.optimizationParameters,
            'model_state_dict': self.net.state_dict()
        }
        torch.save(savedModel, path)

    def plotLosses(self):
        pass

def testDeadAliveNet():
    pass