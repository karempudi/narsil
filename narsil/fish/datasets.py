from scipy.ndimage.morphology import distance_transform_edt
import torch
import torch.nn.functional as F
import numpy as np
import glob
import os
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
from narsil.utils.growth import exp_growth_fit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class singleChannelFishData(object):
	"""
	Class that bundles the FISH data from all the imaging channels and
	generate bounding boxes around the signal and also does background
	subtraction

    Arguments:
    channelNames = list of channel names eg. ['488', 'cy3', 'cy5', 'txr']
    transforms = type of transform you want applied on the fluor images
    fluorChThreshold = list of values for thresholding individual flourChannels
    stdThreshold = no of deviations above which the the flour signal is thresholded in
                   mean-std-normalization
    
    """
    def __init__(self, fishDirName, channelNames, channelSize = (60, 750), minboxHeight = 90, transforms = None, 
                    fluorChThreshold = None, stdThreshold = 1.0, fileFormat = '.tiff',
                    backgroundFishData = None):
        self.fishDirName = fishDirName
        self.channelNames = channelNames
        self.fileFormat = fileFormat
        self.stdThreshold = stdThreshold
        self.fishImages = {}
        self.fishBoxes = {}
        self.fluorChThreshold = fluorChThreshold
        self.channelSize = channelSize
        self.transforms = transforms
        self.minboxHeight = minboxHeight

        # pass in a data object of class singleChannelFishData of the background
        # channel for subtraction
        self.backgroundFishData = backgroundFishData
        for channel in channelNames:
            image = imread(self.fishDirName + channel + self.fileFormat, as_gray=True)
            #image = (image - np.min(image))/(np.max(image) - np.min(image)) * 65535
            if transforms == 'mean-std-normalization':
                self.fishImages[channel]  = (image - np.mean(image))/np.std(image) > stdThreshold

            # basic idea to be able to draw a box around the region where the flour channels are
            elif transforms == 'box':
                #image = gaussian_filter(image, sigma = 4)
                #background subtraction
                image = image - self.backgroundFishData[channel]
                image = gaussian_filter(image, sigma = 2)
                #print("Image found")
                if np.sum(image) == 0:
                    self.fishImages[channel] = self.backgroundFishData[channel]
                    self.fishBoxes[channel] = []
                else:
                    image[:self.channelSize[0], :] = 0
                    image[self.channelSize[1]:, :] = 0
                    #image = (image - np.min(image))/(np.max(image) - np.min(image)) * 65535
                    #image = image - np.mean(image)
                    image_bool = image > max(threshold_otsu(image), self.fluorChThreshold[channel])
                    boxes = []
                    y1 = 5
                    y2 = image.shape[1] - 5
                    width = y2 - y1 + 1
                    xlims_bool = np.sum(image_bool, axis=1) > 0
                    xlims = np.where(np.diff(xlims_bool) == 1)[0]
                    #print(f"{channel}: --> xlims: {xlims}")
                    if len(xlims)%2 == 1 and len(xlims) != 0:
                        xlims = xlims[:-1]
                    for i in range(0, len(xlims), 2):
                        xy = (y1, xlims[i])
                        height = xlims[i+1] - xlims[i] + 1
                        if height >= self.minboxHeight:
                            boxes.append((xy, width, height))
                    #print(f"{channel} : {boxes}")
                    if len(boxes) != 0:
                        self.fishBoxes[channel] = boxes
                    else:
                        self.fishBoxes[channel] = []
                    self.fishImages[channel] = image

            else:
                self.fishImages[channel] = image

        # kinda apply transforms to clean up individual flourescent channels and 
        # make them binary masks to mapy correctly

    def __len__(self):
        return len(self.channelNames)

    def __getitem__(self, channel):
        if channel not in self.channelNames:
            return None
        else:
            return self.fishImages[channel]

    def plot(self, channel):
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        ax.imshow(self.fishImages[channel], cmap='gray')
        if channel in self.fishBoxes.keys():
            for box in self.fishBoxes[channel]:
                ax.add_patch(Rectangle(*box, linewidth = 1, edgecolor='r', facecolor ='none'))
        ax.set_title(f'{channel} Image ')
        plt.show()

    # This will count the number of channels in which the boxes are drawn 
    # around the fluor pixels in the images.
    def getNumberFluorChannels(self):
        numberChannels = 0
        for channel in self.channelNames:
            if len(self.fishBoxes[channel]) > 0:
                numberChannels += 1
        return numberChannels

    def getFluorChannels(self):
        fluorChannels = []
        for channel in self.channelNames:
            if len(self.fishBoxes[channel]) > 0:
                fluorChannels.append(channel)
        return fluorChannels

    def plotAllChannels(self):
        fig, ax = plt.subplots(nrows = 1, ncols = len(self.channelNames))
        for i, channel in enumerate(self.channelNames, 0):
            ax[i].imshow(self.fishImages[channel], cmap='gray')
            if self.transforms == 'box' and len(self.fishBoxes[channel]) != 0:
                for box in self.fishBoxes[channel]:
                    ax[i].add_patch(Rectangle(*box, linewidth = 1, edgecolor = 'r', facecolor = 'none'))
            ax[i].set(title=channel)
        plt.show(block = False)
        fig.canvas.set_window_title(f"{self.fishDirName}")


class singlePositionFishData(object):
    """
    Class for holding fluorescent image of one position and also helps in 
    a background subtraction and extraction of individual channels from 
    channel locations

    """
    
    def __init__(self, fishDir, channelNames, locations, 
            channelWidth=40, imgName='img_000000000', fileformat='.tiff'):
        self.fishDir = fishDir
        self.channelNames = channelNames
        self.locations = locations #locations of the mother machine channels
        self.channelWidth = channelWidth
        self.fileformat = fileformat
        self.fishImages = {}

        # set images
        for channel in self.channelNames:
            image = imread(fishDir + '/' + str(channel) + '/' + imgName + self.fileformat, as_gray=True)
            self.fishImages[channel] = image

    def __len__(self):
        return len(self.channelNames)
    
    def __getitem__(self, channel):
        if channel not in self.channelNames:
            return None
        else:
            return self.fishImages[channel]
    
    def generateBboxes(self):
        pass
    
    def subtractBackground(self):
        pass

    def plotChannel(self, channel, withBboxes=True):
        pass
