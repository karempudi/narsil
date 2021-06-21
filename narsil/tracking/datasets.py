# Datasets loaders for tracking package,including training and testing and running the tracker
import glob
import torch
import numpy as np
from collections import OrderedDict
from skimage.measure import regionprops, label
from skimage.io import imread
from skimage.transform impor resize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from narsil.utils.growth import exp_growth_fit
from narsil.utils.transforms import singleBlobTensorize
from narsil.tracking.datasets import oneCellLineage
from narsil.fish.datasets import singleChannelFishData

np.seterr(over='ignore')



class oneCellLineage(object):
    """
    This class keeps track of one track, starting from birth to division
    and also attributes that help us locate its daughter tracks 
    """

    def __init__(self, dirName,exptTime = 61, fileformat = '.tiff'):
        self.dirName = dirName
        self.fileformat = fileformat
        # Tracks dictionary is something like (str) 'fileName': (int) blob number
        self.trackDictionary = OrderedDict()
        # is also a dictionary with something like (str) 'fileName': (int) blob number
        self.daughtersDictionary = OrderedDict()
        # is also a dicitonary with something like (str) 'fileName': (int) blob number

        self.fileBlobsDict = OrderedDict()
        self.parent = {}

        self.species = None
        self.fluorChannels = None
        # properties to grab areas or lengths
        self.props = {}
        self.areas = {}
        self.lengths = {}

        self.trackLength = 0
        self.exptTime = exptTime

        self.areas_np = -1 * np.ones((exptTime,))
        self.growth_np = -1 * np.ones((exptTime,))
        self.areas_rolling_np = -1 * np.ones((exptTime,))
        self.growth_rolling_np = -1 * np.ones((exptTime,))


        # These get set as internals and need to be reset if you by chance delete an
        # element form the list of tracks in one single channel.
        # These variables are set to link with daughters or parents and useful for 
        # iteration elsewhere
        self._indexToDaughters = []
        

    def __str__(self):
        # Here you print a representation of the lineage, tracks linked and
        # also daughters
        printStr = ""
        printStr += "Printing Track filename and Blob number\n"
        printStr += "---------------------------------------\n"
        if len(self.trackDictionary) > 0:
            for key, value in self.trackDictionary.items():
                printStr += (str(key) + " ----> " + str(value) + "\n")
        else:
            printStr += "No blobs linked"

        printStr += "---------------------------------------\n"
        printStr += "Printing daughter tracks:\n"
        if len(self.daughtersDictionary) > 0:
            for key, value in self.daughtersDictionary.items():
                printStr += (str(key) + " ----> " + str(value) + "\n")
        else:
            printStr += "No daughters linked\n"
        
        printStr += "--------------------------------------\n"

        printStr += "Areas:\n"
        if len(self.areas) > 0:
            for key, value in self.areas.items():
                printStr += (str(key) + '---->' + str(value) + "\n")
        else:
            printStr += "No areas found\n"
        printStr += "-------------------------------------\n"

        printStr += "Species: " + str(self.species) + "\n"

        printStr += "Track Length: " + str(self.trackLength) + "\n"

        printStr += "Fluor Channels: " + str(self.fluorChannels) + "\n"
        
        return printStr

    def __len__(self):
        return self.trackLength
    

    def add(self, fileNumber, blobNumber):
        fileName = self.dirName + str(fileNumber) + self.fileformat
        self.trackDictionary[fileName] = blobNumber
        self.fileBlobsDict[fileNumber] = blobNumber
        self.props[fileName] = regionprops(label(imread(fileName)))[int(blobNumber)]
        self.areas[fileName] = self.props[fileName]['area']
        self.lengths[fileName] = self.props[fileName]['major_axis_length']
        self.trackLength += 1

    def setDaughter(self, fileNumber, blobNumber):
        fileName = self.dirName + str(fileNumber) + self.fileformat
        if fileName in self.daughtersDictionary:
            self.daughtersDictionary[fileName].append(blobNumber)
        else:
            self.daughtersDictionary[fileName] = [blobNumber]
    

    def setParent(self, fileName, blobNumber):
        self.parent[fileName] = blobNumber

    def numDaughters(self):
        return len(self.daughtersDictionary)

    def numParents(self):
        return len(self.parent)




    # Add times form the metadata file somehow, to keep track 
    # of actual times
    def addTimes(self):
        pass

    def setSpecies(self, species):
        self.species = species

    def plotAreas(self):
        areasPlot = []
        for fileName, area in sorted(self.areas.items(), key= lambda kv : int(kv[0].split('/')[-1].split('.')[0])):
            areasPlot.append(area)
        
        # you can add time points loop here later
        plt.figure()
        plt.plot(areasPlot, 'r-')
        plt.show()

    def averageArea(self):
        areasPlot = []
        for fileName, area in sorted(self.areas.items(), key= lambda kv : int(kv[0].split('/')[-1].split('.')[0])):
            areasPlot.append(area)
        return np.mean(np.asarray(areasPlot))
 
    
    def calculateGrowth(self):
        sortedAreasDict = sorted(self.areas.items(), key= lambda kv : int(kv[0].split('/')[-1].split('.')[0]))
        startTime = int(sortedAreasDict[0][0].split('/')[-1].split('.')[0])
        endTime = int(sortedAreasDict[-1][0].split('/')[-1].split('.')[0])
        time = startTime
        for filename, areas in sortedAreasDict:
            self.areas_np[time] = areas
            time += 1

        for i in range(startTime+1, endTime+1):
            self.growth_np[i] = (self.areas_np[i] - self.areas_np[i-1])/self.areas_np[i-1]

        # Add the daughter calculation here
        # TODO
        
        return self.growth_np

    # Used for plotting the growth of one cell track
    def plotGrowth(self):
        pass

    # Used for splitting the track if it only has one daughter
    def split(self):
        pass


    def rollingGrowthRate(self, width = 5, fitatleast = 3):
        sortedAreasDict = sorted(self.areas.items(), key= lambda kv : int(kv[0].split('/')[-1].split('.')[0]))
        startTime = int(sortedAreasDict[0][0].split('/')[-1].split('.')[0])
        endTime = int(sortedAreasDict[-1][0].split('/')[-1].split('.')[0])
        time = startTime
        for filename, areas in sortedAreasDict:
            self.areas_rolling_np[time] = areas
            time += 1
        #print(self.areas_rolling_np)
        
        for i in range(width-1, len(self.areas_rolling_np)):
            current_areas = self.areas_rolling_np[i - width + 1 :i + 1]
            current_times = np.arange(i - width + 1, i + 1)
            set_areas = np.sum(current_areas != -1)
            #print(f"{current_times} ---> {current_areas} ---> {set_areas}")
            # you would need 3 points atleast to fit an exponential curve, you can filter for more points
            # by changing the width and fitatleast parameters
            # to break out rigth after you leave the track use
            #if set_areas >= fitatleast and current_areas[-1] != -1
            if set_areas >= fitatleast : 
                valid_areas_indices = np.where(current_areas != -1)
                valid_areas = current_areas[valid_areas_indices]
                valid_times = current_times[valid_areas_indices]
                #print(f"Valid: {valid_times} ----> {valid_areas}")
                popt, _ = curve_fit(exp_growth_fit, valid_times - valid_times[0], valid_areas)
                #print(f"Growth Rate {-1 * popt[1]}")
                self.growth_rolling_np[i] = -1 * popt[1]

        return self.growth_rolling_np

    def plotRollingGrowth(self, width = 5, fitatleast = 3, colorMap = colorMap, ylim = (-0.04, 0.06)):
        growth_rates = self.rollingGrowthRate(width = width, fitatleast = fitatleast)
        existing_growth_rates = growth_rates[np.where(growth_rates != -1)]
        time_points = np.where(growth_rates != -1)[0]
        
        plt.figure()
        plt.plot(time_points, existing_growth_rates, str(colorMap[self.species]) + '*')
        plt.xlim(0, self.exptTime)
        plt.ylim(*ylim)
        plt.show()


