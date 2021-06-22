# Datasets loaders for tracking package,including training and testing and running the tracker
import glob
import torch
import numpy as np
from collections import OrderedDict
from skimage.measure import regionprops, label
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from narsil.utils.growth import exp_growth_fit
from narsil.utils.transforms import singleBlobTensorize
from narsil.fish.datasets import singleChannelFishData
from torch.utils.data import Dataset, DataLoader

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

    def getWritableData(self):
        # Pacakge the useful data from the track in a correct format, for compression 
        # and saving and reloading.
        return




class singleChannelTrackingSiamese(Dataset):

    """
    Arguments:
         dirName -- dirname of the segmentation of individual channels
         net -- Net that is loaded in eval mode, will use no_grad() and pass blobs

    """
    def __init__(self, dirName, trackingParameters = None,backgroundFishData=None, net=None, fileformat='.tiff',
            maxCentroidMovement=50, channelSize=(150, 850), fishTransforms='box',
            fluorChThreshold=None, flourStdThreshold = 1.0, minFlourBoxHeight=90):

        self.dirName = dirName
        self.fileformat = fileformat
        self.siameseNet = net
        self.imgStdSize = (96, 32)
        self.maxCentroidMovement = maxCentroidMovement
        self.threshold = trackingParameters['threshold']
        self.differenceThreshold = trackingParameters['differenceThreshold']
        self.lastBlobCentroidCutoff = trackingParameters['lastBlobCentroidCutoff']
        self.divisionAreaRatio = trackingParameters['divisionAreaRatio']
        self.maxAreaGrowth = trackingParameters['maxAreaGrowth']
        self.exptTime = trackingParameters['exptTime']
        self.speciesMap = trackingParameters['speciesMap']
        self.numFluors = trackingParameters['numFluors']
        self.doneTracking = False

        self.transform = singleBlobTensorize()

        self.fileindices = [int(filename.split('.')[0].split('/')[-1]) for filename in 
                        glob.glob(self.dirName + "*" + self.fileformat)]

        self.fileindices.sort()
        self.nImages = len(self.fileindices)

        self.filenames = []
        self.images = []
        self.properties = []
        self.nBlobsPerImage = []

        for i in range(self.nImages):
            filename = self.dirName + str(self.fileindices[i]) + self.fileformat
            image = imread(filename) 
            props = regionprops(label(image))
            self.filenames.append(filename)
            self.nBlobsPerImage.append(len(props))
            self.images.append(image)
            self.properties.append(props)
        
        # data for net
        self.dataForNet = []

        # set fish data
        self.fishDirectory = self.dirName.split('blobs')[0] + 'fishChannels' + self.dirName.split('blobs')[-1]
        self.fishdata = singleChannelFishData(self.fishDirectory, backgroundFishData.channelNames,
                    channelSize=channelSize, transforms=fishTransforms, fluorChThreshold=fluorChThreshold,
                    minboxHeight=minFlourBoxHeight, backgroundFishData=backgroundFishData)
        
        if self.fishdata.getNumberFluorChannels() <= self.numFluors:
            self.doneTracking = True
            self.createDataForNet()

        self.speciesNames = None

    
    def createDataForNet(self):
        for fileIndex in range(len(self.filenames) - 1):
            prevfilename = self.filenames[fileIndex]
            currfilename = self.filenames[fileIndex + 1]
            # No loop throught each filename and 
            # get the blobs from previous and current filename and do some transforms
            # and then add the bundles to the dataset
            for blob1index in range(self.nBlobsPerImage[fileIndex]):
                blob1 = self.transform(self.getBlobForNet(fileIndex, blob1index))
                for blob2index in range(self.nBlobsPerImage[fileIndex + 1]):
                    blob2 = self.transform(self.getBlobForNet(fileIndex + 1, blob2index))

                    #if abs(blob1['props'][5].item() - blob2['props'][5].item()) < self.maxCentroidMovement:
                    self.dataForNet.append((blob1['props'], blob1['image'], blob2['props'], blob2['image'],
                                prevfilename, blob1index, currfilename, blob2index))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.dataForNet[idx]

    def __len__(self):
        return len(self.dataForNet)


    def getBlobForNet(self, fileIndex, blobIndex):
        blobProperties = []
        blobProperties.extend(list(self.properties[fileIndex][blobIndex]['bbox']))
        blobProperties.append(self.properties[fileIndex][blobIndex]['area'])
        blobProperties.extend(list(self.properties[fileIndex][blobIndex]['centroid']))
        blobProperties.append(self.properties[fileIndex][blobIndex]['eccentricity'])
        blobProperties.append(self.properties[fileIndex][blobIndex]['major_axis_length'])
        blobProperties.append(self.properties[fileIndex][blobIndex]['minor_axis_length'])
        blobProperties.append(self.properties[fileIndex][blobIndex]['perimeter'])
        image = self.properties[fileIndex][blobIndex]['image']
        
        # Do some operations on the image to center the image in the field,
        # to standardize the inputs to the network
        height, width = image.shape
        #print("Image shape before:", image.shape)
        # set blank background image
        height_outside_limit = height > self.imgStdSize[0]
        width_outside_limit = width > self.imgStdSize[1]

        if height_outside_limit:
            image = resize(image, (self.imgStdSize[0], width))
        if width_outside_limit:
            image = resize(image, (height, self.imgStdSize[1]))

        background = np.zeros(self.imgStdSize)

        # recalculate height after rescaling in case
        height, width = image.shape
        if height%2 == 1:
            xlim_1 = self.imgStdSize[0]//2 - height//2
            xlim_2 = self.imgStdSize[0]//2 + height//2 + 1
        elif height%2 == 0:
            xlim_1 = self.imgStdSize[0]//2 - height//2
            xlim_2 = self.imgStdSize[0]//2 + height//2

        if width%2 == 1:
            ylim_1 = self.imgStdSize[1]//2 - width//2
            ylim_2 = self.imgStdSize[1]//2 + width//2 + 1
        elif width%2 == 0:
            ylim_1 = self.imgStdSize[1]//2 - width//2
            ylim_2 = self.imgStdSize[1]//2 + width//2

        #print(xlim_1, xlim_2, ylim_1, ylim_2)
        #print("Image shape:", image.shape)
        
        background[xlim_1 : xlim_2, ylim_1: ylim_2] = image
        return {'props': blobProperties, 'image': background}



    def getFileNumber(self, filename):
        return int(filename.split('.')[0].split('/')[-1])
    # From the net you get scores as output, you will have to construct appropriate links 
    # here before 
    def constructLinks(self, prevfilenames, currfilenames, prevBlobNumbers, currBlobNumbers, netScores):
        self.fileNamesProcessed = []
        self.links = []
        self.scoresArray = np.zeros(shape=(len(prevfilenames), 5))
        for i in range(len(prevfilenames)):
            self.scoresArray[i, 0] = self.getFileNumber(prevfilenames[i]) 
            self.scoresArray[i, 1] = self.getFileNumber(currfilenames[i]) 
            self.scoresArray[i, 2] = prevBlobNumbers[i]
            self.scoresArray[i, 3] = currBlobNumbers[i]
            self.scoresArray[i, 4] = netScores[i][0]

        for fileIndex in range(len(self.filenames) - 1):

            alreadyLinked = [] # trying to avoid linking things already linked

            for blob1index in range(self.nBlobsPerImage[fileIndex]):
                scores = -np.zeros((self.nBlobsPerImage[fileIndex + 1]))
                for blob2index in range(self.nBlobsPerImage[fileIndex + 1]):
                    scores[blob2index] = self.getScore(fileIndex, fileIndex + 1, blob1index, blob2index)
                
                #indicesAfterCutoff = np.where(scores < self.threshold)[0]
                sorted_scores = np.sort(scores)

                #if len(indicesAfterCutoff) == 1 and (indicesAfterCutoff[0] not in alreadLinkedInNext):
                #    self.links.append([fileIndex, fileIndex+1, blob1index, indicesAfterCutoff[0]])
                #   alreadLinkedInNext.append(indicesAfterCutoff[0])
                
                #elif len(indicesAfterCutoff) == 2:
                    # check if both pass the difference threshold
                    #daughter1Index = indicesAfterCutoff[0]
                    #daughter2Index = indicesAfterCutoff[1]

                    #if (abs(scores[daughter1Index] - scores[daughter2Index]) <= self.differenceThreshold):
                    #    # link both if area constraints are met
                    #    if ((self.properties[fileIndex+1][daughter1Index]['area'] < self.divisionAreaRatio * self.properties[fileIndex][blob1index]['area'])
                    #        and (self.properties[fileIndex+1][daughter2Index]['area'] < self.divisionAreaRatio * self.properties[fileIndex][blob1index]['area'])):
                    #        if (daughter1Index not in alreadLinkedInNext):
                    #            self.links.append([fileIndex, fileIndex+1, blob1index, daughter1Index])
                    #            alreadLinkedInNext.append(daughter1Index)
                    #        if (daughter2Index not in alreadLinkedInNext):
                    #            self.links.append([fileIndex, fileIndex + 1, blob1index, daughter2Index])
                    #            alreadLinkedInNext.append(daughter2Index)
                    #    else:
                    #        linkedblobIndex = daughter1Index
                    #        if daughter1Index not in alreadLinkedInNext:
                    #            self.links.append([fileIndex, fileIndex+1, blob1index, linkedblobIndex])
                    #            alreadLinkedInNext.append(linkedblobIndex)
                    #        elif (linkedblobIndex in alreadLinkedInNext):
                    #            linkedblobIndex = daughter2Index
                    #            self.links.append([fileIndex, fileIndex+1, blob1index, linkedblobIndex])
                    #            alreadLinkedInNext.append(linkedblobIndex)
           
                if len(sorted_scores) == 1:
                # check if it is less than threshold and add up to the links
                # and also the centroid of teh blob2 is not past the cutoff
                    if(sorted_scores <= self.threshold and
                    self.properties[fileIndex+1][blob2index]['centroid'][0] < self.lastBlobCentroidCutoff):
                        self.links.append([fileIndex, fileIndex+1, blob1index, blob2index])
                        alreadyLinked.append(blob2index)
                    
                elif len(sorted_scores) >= 2:
                    # check the difference between the first elements and if they pass the threshold
                    # find the indiecs of thes two elements as they give the blob index

                    if((sorted_scores[0] <= self.threshold) and (sorted_scores[1] <= self.threshold)):
                        # both are less than threshold, check the difference and add links
                        # They are daughters, find the blob indices in the original array scores and
                        # add them to the links
                        daughterIndex1 = np.where(scores == sorted_scores[0])[0][0]
                        daughterIndex2 = np.where(scores == sorted_scores[1])[0][0]
                        if(abs(sorted_scores[0]  - sorted_scores[1]) <= self.differenceThreshold):
                            # check for area constrains and alrad linked or not constraint
                            if((self.properties[fileIndex+1][daughterIndex1]['area'] < self.divisionAreaRatio * self.properties[fileIndex][blob1index]['area'])
                            and (self.properties[fileIndex+1][daughterIndex2]['area'] < self.divisionAreaRatio * self.properties[fileIndex][blob1index]['area'])):
                                if (daughterIndex1 not in alreadyLinked):
                                    self.links.append([fileIndex, fileIndex + 1, blob1index, daughterIndex1])
                                    alreadyLinked.append(daughterIndex1)
                                if (daughterIndex2 not in alreadyLinked):
                                    self.links.append([fileIndex, fileIndex + 1, blob1index, daughterIndex2])
                                    alreadyLinked.append(daughterIndex2)
                            else:
                                linkedblobIndex = daughterIndex1
                                if(linkedblobIndex not in alreadyLinked):
                                    self.links.append([fileIndex, fileIndex+1, blob1index, linkedblobIndex])
                                    alreadyLinked.append(linkedblobIndex)
                                elif(linkedblobIndex in alreadyLinked):
                                    linkedblobIndex = daughterIndex2
                                    self.links.append([fileIndex , fileIndex+1, blob1index, linkedblobIndex])
                                    alreadyLinked.append(linkedblobIndex)
                        
                        else:
                            # They don't reach the differene threshold
                            # so only link the closed blob, if it is not linked already, if linked, go
                            # to the next blob
                            linkedblobIndex = daughterIndex1
                            if (linkedblobIndex not in alreadyLinked):
                                self.links.append([fileIndex, fileIndex +1, blob1index, linkedblobIndex])
                                alreadyLinked.append(linkedblobIndex)
                            elif (linkedblobIndex in alreadyLinked):
                                linkedblobIndex = daughterIndex2
                                self.links.append([fileIndex, fileIndex + 1, blob1index, linkedblobIndex])
                                alreadyLinked.append(linkedblobIndex)
                    
                    # only one score is less than threshold
                    elif (sorted_scores[0] <= self.threshold and sorted_scores[1] > self.threshold):
                        # Just the least scores is less than threshold
                        linkedblobIndex = np.where(scores == sorted_scores[0])[0][0]
                        if (linkedblobIndex not in alreadyLinked):
                            if(self.properties[fileIndex+1][linkedblobIndex]['centroid'][0] < self.lastBlobCentroidCutoff):
                                self.links.append([fileIndex, fileIndex+1, blob1index, linkedblobIndex])
                                alreadyLinked.append(linkedblobIndex)
                    
                    
                
            self.fileNamesProcessed.append(self.dirName + str(fileIndex) + self.fileformat)

    # Look in side the lines of score structure and spit out the float value of the
    # score with the currentFileaname, nextFileName, blob1index, blob2index
    def getScore(self, currentFileIndex, nextFileIndex, blob1index, blob2index):
        #linksArray = np.asarray(self.scoresStructure)
        indices = np.where((self.scoresArray[:, :4] == [currentFileIndex, nextFileIndex, blob1index, blob2index]).all(axis = 1))[0]
        if len(indices) == 0:
            return -1.0 #if no match return a large value instead
        else:
            return float(self.scoresArray[indices][0][4])
                

    # convert the links made from scores, to convert to track structures
    def convertLinksToTracks(self):

        self.tracks = []
        # Here you will have to do the clean up as well
        
        # In this function you will have to construct the lineages of one cell and then
        # add it to the list of tracks

        # TODO: Here is where you construct tracks

        # TODO: Needs a bit more sophistication depending on how good 
        # the track links are
        # This is a function that constructs tracks from the links which 
        # is a array of list: [file1, file2, blob1, blob2].
        # Iterate till you go no where or have two blobs connected to one blob
        # 
        #self.tracks = [] # a list of dictionaries, each track is a dictionary
        # keys are frame numbers and values are indices of the blobs
        # create local copies tot not change the classes internal ones
        linksArray = np.asarray(self.links)
        
        while linksArray.size != 0:
        #for row in range(linksArray.shape[0]):
            # start with one element and end when the track ends
            # or when there are two blobs connecting the same blob
            first_row = linksArray[0] # start linking from here
            one_track = oneCellLineage(self.dirName, exptTime= self.exptTime)
            one_track.add(first_row[0], first_row[2]) # add the first blob
            # Now loop till there is nothing matching
            # How many blobs is this blob connected to : this blob means the first row
            while True:
                current_blob_index = np.where((linksArray[:, 0::2] == [first_row[0], first_row[2]]).all(axis=1))[0]
                #print("Current_blob_index: ", current_blob_index)
                connections_from_current_blob = np.where((linksArray[:, 0::2] == [first_row[0], first_row[2]]).all(axis=1))[0]
                if(len(connections_from_current_blob) == 2):
                    # Time to split the track and not add the track, find out which are daughters and keep track
                    #print("Blob has daughter and split: ", first_row, connections_from_current_blob)
                    daughterIndex1 = connections_from_current_blob[0]
                    daughterIndex2 = connections_from_current_blob[1]
                    one_track.setDaughter(linksArray[daughterIndex1][1], linksArray[daughterIndex1][3])
                    #print(linksArray[daughterIndex1][1], linksArray[daughterIndex1][3])
                    #print(linksArray[daughterIndex2][1], linksArray[daughterIndex2][3])
                    one_track.setDaughter(linksArray[daughterIndex2][1], linksArray[daughterIndex2][3])
                    linksArray = np.delete(linksArray, (current_blob_index), axis = 0)
                    break
                elif(len(connections_from_current_blob) == 1):
                    area_ratio = self.properties[first_row[1]][int(first_row[3])]['area']/self.properties[first_row[0]][int(first_row[2])]['area']
                    #print(area_ratio)
                    # link only if the area ratio falls in certain range
                    if area_ratio > 0.7 and area_ratio < (1/0.7):
                        one_track.add(first_row[1], first_row[3])
                        linksArray = np.delete(linksArray, (current_blob_index), axis = 0)
                        next_blob_index = np.where((linksArray[:,0::2] == [first_row[1], first_row[3]]).all(axis=1))[0] # Grab the index of the next blob from the array
                        if(next_blob_index.size != 0):
                            first_row = linksArray[next_blob_index[0]]
                    # one daughter case
                    elif area_ratio < 0.7 and area_ratio > 0.35:
                        daughterIndex1 = connections_from_current_blob[0]
                        one_track.setDaughter(linksArray[daughterIndex1][1], linksArray[daughterIndex1][3])
                        linksArray = np.delete(linksArray, (current_blob_index), axis = 0)
                        break
                    # don't link anything if area ration is off
                    else:
                        linksArray = np.delete(linksArray, (current_blob_index), axis = 0)
                        break

                elif(len(connections_from_current_blob) == 0):
                    break
            #print(one_track)
            self.tracks.append(one_track)
        



    def labelTracksWithFluorChannels(self, maxIterations=10, printFluorLabels=False):
        if self.fishdata == None:
            print("Fish linking failed")
            return None

        #print(f"Labeling some of the {len(self.tracks)}")

        # label all thracks that end in the last frame first
        lastFileName = self.fileNamesProcessed[-1]
        for cellLineage in self.tracks:
            # check if the track ends in the lastFrame
            if lastFileName in cellLineage.trackDictionary:
                # label track with correct flour channels to each of the track
                centroid = cellLineage.props[lastFileName]['centroid']
                cellLineage.fluorChannels = self.getFluorChannels(centroid)
                #print(cellLineage.fluorChannels)
        
        # Loop over and set the internal daughter indices of each track in the set of tracks
        self.setDaughterIndices()
        #print("Set internal daughter indices")
        while(maxIterations > 0):
            for _, oneLineage in enumerate(self.tracks, 0):
                indices = oneLineage._indexToDaughters
                #print(f"{indices} ----> {oneLineage.fluorChannels}")
                if oneLineage.fluorChannels == None and len(indices) > 0:
                    # set it to one of the daughters
                    for index in indices:
                        oneLineage.fluorChannels = self.tracks[index].fluorChannels
                        if oneLineage.fluorChannels != None:
                            break
            maxIterations -= 1

        # Loop over and check if the tracks's and their daughter species mathc
        # If not label it as conflict

        for _, oneLineage in enumerate(self.tracks, 0):
            daughterIndices = oneLineage._indexToDaughters
            for index in indices:
                if oneLineage.fluorChannels != self.tracks[index].fluorChannels:
                    oneLineage.fluorChannels = ['Conflict']
                    break
        if printFluorLabels == True:
            for oneLineage in self.tracks:
                print(oneLineage.fluorChannels)
    

    # Loop over the set of tracks and set the internal indices of the lineages in the overall
    # tracks list
    def setDaughterIndices(self):

        for index, oneLineage in enumerate(self.tracks, 0):
            oneLineage._indexToDaughters = []
            if oneLineage.numDaughters() > 0:
                # get daughter track's index in the set of tracks
                
                daughterStartFileName = list(oneLineage.daughtersDictionary.keys())[0]
                daughterBlobIndices = oneLineage.daughtersDictionary[daughterStartFileName]
                #print(f"{daughterStartFileName} ---> {daughterBlobIndices}")

                # Now loop over and set internal indices of the single track variables
                for index1, oneLineage1 in enumerate(self.tracks, 0):
                    if daughterStartFileName in oneLineage1.trackDictionary and oneLineage1.trackDictionary[daughterStartFileName] in daughterBlobIndices:
                        oneLineage._indexToDaughters.append(index1)

    # Function that takes in a species map and goes through all the tracks and 
    # assigns species labels.
    # Species Map is a mapping from the fluorescence channels to the name of the species
    def setSpeciesForAllTracks(self):
        self.speciesNames = []
        for species in self.speciesMap:
            self.speciesNames.append(species)
        
        for oneLineage in self.tracks:
            if oneLineage.fluorChannels != None:
                for species, fluorChannels in self.speciesMap.items():
                    if fluorChannels == set(oneLineage.fluorChannels):
                        #set species
                        oneLineage.species = species
                        #print(oneLineage.species)
                        break


    def insideBox(self, boxes, centroid):
        # Checking only x coordinate and forgetting about y coordinate of the boxes for now.
        # TODO: check y corrdinate later # might be useful when you have errors in chip making
        for box in boxes:
            if centroid[0] >= box[0][1] and centroid[0] <= box[0][1] + box[2]:
                return True

        return False

    def getFluorChannels(self, centroid):
        # loop over the fluor channels and return a list of all the channels in which
        # the centroid lies
        fluorChannels = []
        for channel in self.fishdata.channelNames:
            if self.insideBox(self.fishdata.fishBoxes[channel], centroid):
                fluorChannels.append(channel)
        
        if len(fluorChannels) == 0:
            return None
        else:
            return fluorChannels
        

    def calculateGrowth(self, minTrackLength = 5, minArea=100):
        channelGrowth = []
        tracks = self.convertLinksToTracks()
        for track in tracks:
            if len(track) > minTrackLength and track.averageArea() > minArea:
                channelGrowth.append(track.calculateGrowth())
        
        growthRates = np.asarray(channelGrowth)
        ntracks, ntimepoints = growthRates.shape
        mean_growth = np.zeros((ntimepoints,))
        for timepoint in range(ntimepoints):
            growth_t = growthRates[:, timepoint]
            set_growth = growth_t[np.where(growth_t != -1)]
            if set_growth.size != 0:
                mean_growth[timepoint] = np.mean(set_growth)

        return mean_growth, growthRates


    def getGrowthOfSpeciesLabeledTracks(self, width=5, fitatleast=3):
        growthRates = {}
        for species in self.speciesNames:
            growthRates[species] = []

        for i, track in enumerate(self.tracks, 0):
            if track.species != None:
                growth = track.rollingGrowthRate(width = width, fitatleast = fitatleast)
                growthRates[track.species].append(growth)
                #print(f"Track number --> {i}")
        return growthRates


    # TODO: Add more constraints and filters if needed like you did in 
    # calculateGrowth function like minAverage are and min Track Length (This will be
    # taken care of in the rolling case by other parameters like width and fitatleast)
    def calculateRollingGrowthRates(self, width=5, fitatleast=3):
        channelGrowth = dict()
        for species in self.speciesNames:
            channelGrowth[species] = []
        
        for track in self.tracks:
            if track.species != None:
                channelGrowth[track.species].append(track.rollingGrowthRate(width=width, fitatleast=fitalteast))

        for species in channelGrowth:
            channelGrowth[species] = np.asarray(channelGrowth[species])
        
        return channelGrowth


    def plotAllTracks(self, spacing=0):
        # Call this function only after calling trackFullDir() --> can fix later 
        # to run in loop dynamically
        # Function plots all the track links read from links list
        unlabeleddata = singleChannelDataset(self.dirName, fileformat=self.fileFormat)
        
        props = [] 
        for i in range(len(unlabeleddata)):
            regionpropsOneImage = regionprops(label(unlabeleddata[i]))
            props.append(regionpropsOneImage)

        n_images = len(unlabeleddata)
        print("Number of images: ", n_images)
        if n_images > 0: 
            height, width = unlabeleddata[0].shape
            print("Height: ", height, " Width: ", width)
        full_img = np.zeros(shape=(height, n_images * (width + spacing)))
        #indices = unlabeleddata.indices
        print("Full image shape: ", full_img.shape)

        # set the images
        for i in range(n_images):
            sub_image = unlabeleddata[i]
            spacing_image = np.zeros(shape=(height, spacing))
            sub_image = np.append(sub_image, spacing_image, axis = 1)
            full_img[:, i*(width + spacing) : (i+1) * (width+spacing)] = sub_image

        plt.figure()
        plt.imshow(full_img)

        # From the self.links generated after
        # 

        for row in self.links:
            # get frame_t from links
            frame_t = int(row[0].split('.')[0].split('/')[-1])
            blob_t, blob_t1 = row[2], row[3]
            #print(row)
            centroid_t_x, centroid_t_y = props[frame_t][blob_t]['centroid']
            centroid_t1_x, centroid_t1_y = props[frame_t + 1][blob_t1]['centroid']
            plt.plot([centroid_t_y + frame_t*(width + spacing), centroid_t1_y + (frame_t + 1)*(width + spacing)], [centroid_t_x, centroid_t1_x], 'r')

        plt.title('Links from tracking algorithm')
        plt.show(block = False)
    
    def getLinksBetweenTwoFrames(self, frame_i, frame_j):
        links = []
        for row in self.links:
            if self.getFrameNumber(row[0]) == frame_i and self.getFrameNumber(row[1]) == frame_j:
                links.append(row)

        return links
    
    def getFrameNumber(self, filename):
        return int(filename.split('.')[0].split('/')[-1])


    def plotAllTracksWithFISH(self, spacing=0):
               # Call this function only after calling trackFullDir() --> can fix later 
        # to run in loop dynamically
        # Function plots all the track links read from links list
        if self.fishdata == None:
            print("Fish data is not added")
            return
        
        #print("Number of images: ", n_images)

        if self.nImages > 0: 
            height, width = self.images[0].shape
            #print("Height: ", height, " Width: ", width)

        full_img = np.zeros(shape=(height, self.nImages * (width + spacing)))
        #indices = unlabeleddata.indices
        #print("Full image shape: ", full_img.shape)

        # set the images
        for i in range(self.nImages):
            sub_image = self.images[i]
            spacing_image = np.zeros(shape=(height, spacing))
            sub_image = np.append(sub_image, spacing_image, axis = 1)
            full_img[:, i*(width + spacing) : (i+1) * (width+spacing)] = sub_image

        fig, ax = plt.subplots(nrows=1, ncols = len(self.fishdata) + 1, gridspec_kw={'width_ratios':[self.nImages] + [1 for _ in range(len(self.fishdata))]})
        ax[0].imshow(full_img)

        # From the self.links generated after calling trackFullDir
        # 
        for row in self.links:
            # get frame_t from links
            frame_t = row[0]
            blob_t, blob_t1 = row[2], row[3]
            centroid_t_x, centroid_t_y = self.properties[frame_t][blob_t]['centroid']
            centroid_t1_x, centroid_t1_y = self.properties[frame_t + 1][blob_t1]['centroid']
            ax[0].plot([centroid_t_y + frame_t*(width + spacing), centroid_t1_y + (frame_t + 1)*(width + spacing)], [centroid_t_x, centroid_t1_x], 'r')

        ax[0].set(title ='Links from Siamese tracking algorithm -- Remembe these are links - Not final tracks')

        for i, channel in enumerate(self.fishdata.channelNames, 1):
            ax[i].imshow(self.fishdata[channel], cmap='gray')
            if self.fishdata.transforms == 'box' and len(self.fishdata.fishBoxes[channel]) != 0:
                for box in self.fishdata.fishBoxes[channel]:
                    ax[i].add_patch(Rectangle(*box, linewidth = 1, edgecolor='r', facecolor ='none'))
            ax[i].set(title=channel)
        plt.show(block = False)
    

    def plotTracksUsedForGrowth(self, spacing=0,
                colorMap={'Klebsiella': 'r', 'E.coli': 'b', 'Pseudomonas': 'g', 'E.cocci' : 'm', 'NoTrack': 'c'}):

        if self.fishdata == None:
            print("Fish data is not added")

        if self.nImages > 0:
            height, width = self.images[0].shape

        full_img = np.zeros(shape=(height, self.nImages * (width + spacing)))
        for i in range(self.nImages):
            sub_image = self.images[i]
            spacing_image = np.zeros(shape=(height, spacing))
            sub_image = np.append(sub_image, spacing_image, axis = 1)
            full_img[:, i * (width + spacing) : (i + 1) * (width + spacing)] = sub_image

        #Plot all the images
        fig, ax = plt.subplots(nrows=1, ncols = len(self.fishdata) + 1, gridspec_kw={'width_ratios': [self.nImages] + [1 for _ in range(len(self.fishdata))]})
        ax[0].imshow(full_img, cmap='gray')

        # Plot all the fish channels data and boxes around the thresholded data 
        for i, channel in enumerate(self.fishdata.channelNames, 1):
            ax[i].imshow(self.fishdata[channel], cmap='gray')
            if self.fishdata.transforms == 'box' and len(self.fishdata.fishBoxes[channel]) != 0:
                for box in self.fishdata.fishBoxes[channel]:
                    ax[i].add_patch(Rectangle(*box, linewidth = 1, edgecolor='r', facecolor ='none'))
            ax[i].set(title=channel)
 
        # for each track in the set of tracks start plotting on the image
        for cellLineage in self.tracks:
            species = cellLineage.species
            if species == None:
                color = colorMap['NoTrack']
            else:
                color = colorMap[species]
            
            # start connecting centroids
            trackAndBlobFiles = list(cellLineage.fileBlobsDict)
            for i in range(len(trackAndBlobFiles) - 1):
                frame_t = trackAndBlobFiles[i]
                frame_t1 = trackAndBlobFiles[i+1]
                blob_frame_t = cellLineage.fileBlobsDict[frame_t]
                blob_frame_t1 = cellLineage.fileBlobsDict[frame_t1]
                centroid_t_x, centroid_t_y = self.properties[frame_t][blob_frame_t]['centroid']
                centroid_t1_x , centroid_t1_y = self.properties[frame_t1][blob_frame_t1]['centroid']
                ax[0].plot([centroid_t_y + frame_t*(width + spacing), centroid_t1_y + (frame_t + 1)*(width + spacing)], [centroid_t_x, centroid_t1_x], color)
        
        plt.show(block=False)
