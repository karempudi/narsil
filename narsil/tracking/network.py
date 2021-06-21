# Network architecture of the nets used for tracking
import torch
import torch.nn as nn
import torch.nn.functional as F


class siameseNet(nn.Module):

    def __init__(self, outputFeatureSize):
        super(siameseNet, self).__init__()
        # Take 11 input properties and then make 10 vector
        self.outputFeatureSize = outputFeatureSize
        self.propertiesHead = nn.Sequential(
            nn.Linear(11, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, self.outputFeatureSize)
        )
        # Do convolutions and then fully connected layer and 
        # make a 10 vector
        # image shape is set in the dataLoader, not here 
        # imgStdsize = (96,32) 
        self.imageHead = nn.Sequential(
                            nn.Conv2d(1, 4, kernel_size=(3,3), stride = 1, padding = 1),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm2d(4),
                            nn.Dropout2d(p=0.2),

                            nn.Conv2d(4, 8, kernel_size=(3,3), stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm2d(8),
                            nn.Dropout2d(p=0.2),

                            nn.Conv2d(8, 8, kernel_size=(3,3), stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm2d(8),
                            nn.Dropout2d(p=0.2),

                            )
        # substitute the * with image dimension
        self.fc1imageHead = nn.Sequential(
                                nn.Linear(8 * 96 * 32, 500),
                                nn.ReLU(inplace=True),
                                nn.Linear(500, 500),
                                nn.ReLU(inplace=True),
                                nn.Linear(500, self.outputFeatureSize)
                            )

        
    # Applies the two network heads and concatenates the output
    def forward_oneSet(self, prop, image):
        propertiesHeadOutput = self.propertiesHead(prop)
        imghead = self.imageHead(image)
        fc1input = imghead.view(imghead.size()[0], -1)
        imageHeadOutput = self.fc1imageHead(fc1input)
        return torch.cat((propertiesHeadOutput, imageHeadOutput), 1) # Careful here, this is tested to work
    
    # This will take two inputs and spit out one output
    # which is something like a distance metric, in the 
    # space of similartiy or not.
    def forward(self, prop1, image1, prop2, image2):
        output1 = self.forward_oneSet(prop1, image1)
        output2 = self.forward_oneSet(prop2, image2)
        return output1, output2