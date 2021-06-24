# File containing datasets used for training the siamese net
import glob
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from skimage.transform import resize
from collections import OrderedDict


class channelDataset(object):

    def __init__(self, directory, labelled= False, frameskip=1, fileformat='.tiff'):

        self.directory = directory
        self.fileformat = fileformat
        self.labelled = labelled
        self.n_images = len(glob.glob(self.directory + "*" + self.fileformat))
        self.indices = [int(filename.split('.')[0].split('/')[-1]) for filename in 
                        glob.glob(self.directory + "*" + self.fileformat)]
        self.indices.sort()
        self.indices = self.indices[::frameskip]

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):

        channel_img_name =  self.directory + str(self.indices[idx]) + self.fileformat
        channel_img = imread(channel_img_name)
        if self.labelled:
            channel_img = label(channel_img, connectivity=2) 
        return channel_img


class siameseDatasetWrapper(Dataset):

    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]

        if self.transforms != None:
            sample = self.transforms(sample)

        return sample

class siameseDataset(object):

    def __init__(self, dirNames, inlcudeDaughters=True, imgStdSize=(96, 32),
                 validation=True,
                 fileformat='.tiff',linksformat='.npy', frameskip=1):
        self.dirNames = dirNames
        self.imgStdSize = imgStdSize
        self.fileformat = fileformat
        self.linksformat = linksformat
        self.frameskip = frameskip
        self.includeDaughters = inlcudeDaughters
        self.validation = validation

        self.data = []

        self.dataInclDaughterLinks = []

        self.trainingData = []
        self.validationData = []

        for directory in self.dirNames:
            tracksSet = setOfTracksTrain(directory, fileformat=self.fileformat,
                                    linksformat=self.linksformat, frameskip=self.frameskip)
            linksFileName = directory + 'justlinks' + self.linksformat
            linksList = np.load(linksFileName).tolist()
            #numberOfFiles = len(tracskSet.indices)
            fileIndices = tracksSet.indices
            #for i in range(len(tracksSet)):
            #    tracksSet.printTrack(i)
            #    print('----------------------')

            # Iterate through the frames and construct data object
            # Better to load them from memory on the fly
            for index in range(len(fileIndices) - 1):
                index_t = fileIndices[index]
                fileName_t = directory + str(index_t) + self.fileformat
                index_t1 = fileIndices[index + 1]
                fileName_t1 = directory + str(index_t1) + self.fileformat
                # Loop through the blobs and add them to data
                imgBundle1 = imgPropBundle(fileName_t, self.imgStdSize)
                imgBundle2 = imgPropBundle(fileName_t1, self.imgStdSize)
                nBlobs1 = len(imgBundle1)
                nBlobs2 = len(imgBundle2)
                # Check the tracksSet elements for index_t: blob_t and index_t1: blob_t1
                # Loop through each set of tracks and check if both exists
                for blob1 in range(nBlobs1):
                    for blob2 in range(nBlobs2):
                        # loop through each of the tracks and check
                        blobs_connected_incl_daughters = False
                        if [index_t, blob1, blob2] in linksList:
                            blobs_connected_incl_daughters = True
                        
                        blobs_connected = False
                        for track in tracksSet:
                            # if both blobs exist in one track, the set the label to 1 as they are linked
                            # if not then set the label to 0, as they are not linked
                            if((index_t in track and blob1 == track[index_t]) and (index_t1 in track and blob2 == track[index_t1])):
                                blobs_connected = True
                                break
                        if (blobs_connected):
                            item = [imgBundle1[blob1], imgBundle2[blob2], True]
                            #print(item)
                            self.data.append(item)
                        elif (not blobs_connected):
                            item = [imgBundle1[blob1], imgBundle2[blob2], False]
                            #print(item)
                            self.data.append(item)
                        if (blobs_connected_incl_daughters):
                            item = [imgBundle1[blob1], imgBundle2[blob2], True]
                            self.dataInclDaughterLinks.append(item)
                        elif (not blobs_connected_incl_daughters):
                            item = [imgBundle1[blob1], imgBundle2[blob2], False]
                            self.dataInclDaughterLinks.append(item)
            
        self.getLinkedData()
        self.getNotLinkedData()
        numberLinkedData = len(self.linkedData)
        self.sampledNonLinkedData = random.sample(self.notLinkedData, numberLinkedData)
        self.balancedData = self.linkedData + self.sampledNonLinkedData
    
    def __len__(self):
        return len(self.balancedData)

    def splitData(self, p=0.8):
        trainData = []
        validationData = []
        if self.validation == True:
            for data in self.balancedData:
                if random.random() >= 1 - p:
                    trainData.append(data)
                else:
                    validationData.append(data)
        else:
            trainData = self.balancedData
        
        return trainData, validationData


    def getLinkedData(self):
        if self.includeDaughters:
            data = self.dataInclDaughterLinks
        else:
            data = self.data
        self.linkedData = []
        for item in data:
            if item[2] == True:
                self.linkedData.append(item)
    
    def getNotLinkedData(self):
        if self.includeDaughters:
            data = self.dataInclDaughterLinks
        else:
            data = self.data
        self.notLinkedData = []
        for item in data:
            if item[2] == False:
                self.notLinkedData.append(item)



class setOfTracksTrain(Dataset):
    """
    A class holding a set of tracks for a channel

    TODO: DONT use frameskip option, it is not fully implemented in all functions, in the current structure 
    """
    def __init__(self, directory, fileformat = '.tiff', linksformat = '.npy',frameskip = 1):
        self.directory = directory
        self.fileformat = fileformat 
        self.frameskip = frameskip
        self.indices = [int(filename.split('/')[-1].split('.')[0]) for filename in glob.glob(self.directory + "*" + self.fileformat)]
        self.indices.sort()
        self.indices = self.indices[::self.frameskip]
        # Data is coming from the labeled dataset
        self.data = channelDataset(self.directory, labelled= True, frameskip=self.frameskip, fileformat=self.fileformat)
        self.unlabeleddata = channelDataset(self.directory, labelled=False, frameskip=self.frameskip, fileformat=self.fileformat)
        self.linksformat = linksformat
        self.linksFileName = self.directory + 'justlinks' + self.linksformat

        # Load the tracks from numpy dictionary here
        self.loadLinksFromNumpy()
        self.props = []
        for i in range(len(self.data)):
            regionpropsOneImage = regionprops(self.data[i])
            self.props.append(regionpropsOneImage)

        # Tracks is a list, could be a set. will refactor later
        # Each element is a track, which is a dictionary containing keys(Frame numbers)
        # and values(blob numbers), and any other details can be added to the values for
        # indexing into later, like quality labels, time stamps, etc
        self.constructTracks()

    def __len__(self):
        # Return the total number of tracks after loading 
        return len(self.tracks)

    def __str__(self):
        return 'Number of tracks are' + str(self.__len__())

    def __getitem__(self, idx):
        # function returns a set of images
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # sort the dictionary items if needed or just in case, 
        sorted_track = OrderedDict(sorted(self.tracks[idx].items()))
        return sorted_track

    def printTrack(self, idx):
        # Funciton that prints the details of one track to terminal to read
        track = self.__getitem__(idx)
        # sort the the dictionary by time and then print the track
        for key, value in track.items():
            print("Frame No: ", key, " Blob No: ", value, "bbox: ", self.props[key][value]['bbox'])
            
    
    def plotTrack(self, idx):
        # Function to plot and track blobs in matplotlib
        track = self.__getitem__(idx)
        track_length = len(track)
        # convert for iteration
        track_list = [(frame_t, blob_t) for frame_t, blob_t in track.items()]
        fig, axs = plt.subplots(1, track_length)
        i = 0
        for frame_t, blob_t in track_list:
            axs[i].imshow(self.props[frame_t][blob_t]['image'])
            #axs[i].title(str(frame_t))
            i += 1
        

    def loadLinksFromNumpy(self):
        # Here iterate over the matrix and figure out the appropriate blobs 
        # You would have to construct the track iteratively as there is no easy way 
        # to access a particular track using a particular blob without looping.
        self.linksArray = np.load(self.linksFileName)

    def plotAllTracksFromNumpy(self, spacing = 0):
        # Function plots all the track links read from numpy training matrix
        n_images = len(self.unlabeleddata)
        print("Number of images: ", n_images)
        if n_images > 0: 
            height, width = self.unlabeleddata[0].shape
            print("Height: ", height, " Width: ", width)
        full_img = np.zeros(shape=(height, n_images * (width + spacing)))
        indices = self.unlabeleddata.indices
        print("Full image shape: ", full_img.shape)

        # set the images
        for i in range(n_images):
            sub_image = self.unlabeleddata[i]
            spacing_image = np.zeros(shape=(height, spacing))
            sub_image = np.append(sub_image, spacing_image, axis = 1)
            full_img[:, i*(width + spacing) : (i+1) * (width+spacing)] = sub_image

        plt.figure()
        plt.imshow(full_img)

        # From the linksArray, grab the stuff from the frame numbers in indices
        # 
        for row_index in range(self.linksArray.shape[0]):
            frame_t, blob_t, blob_t1 = self.linksArray[row_index]
            centroid_t_x, centroid_t_y = self.props[frame_t][blob_t]['centroid']
            centroid_t1_x, centroid_t1_y = self.props[frame_t + 1][blob_t1]['centroid']
            plt.plot([centroid_t_y + frame_t*(width + spacing), centroid_t1_y + (frame_t + 1)*(width + spacing)], [centroid_t_x, centroid_t1_x], 'r')

        plt.title('Links from Numpy traininig matrix')
        plt.show(block = False)
    
    def constructTracks(self):
        # This is a function that constructs tracks from the links Matrix.
        # Iterate till you go no where or have two blobs connected to one blob
        # 
        self.tracks = [] # a list of dictionaries, each track is a dictionary
        # keys are frame numbers and values are indices of the blobs
        # create local copies tot not change the classes internal ones
        linksArray = self.linksArray
        
        while linksArray.size != 0:
        #for row in range(linksArray.shape[0]):
            # start with one element and end when the track ends
            # or when there are two blobs connecting the same blob
            first_row = linksArray[0] # start linking from here
            one_track = {first_row[0] : first_row[1]} # add the first blob
            # Now loop till there is nothing matching
            # How many blobs is this blob connected to : this blob means the first row
            while True:
                current_blob_index = np.where((linksArray[:, :2] == [first_row[0], first_row[1]]).all(axis=1))[0]
                #print("Current_blob_index: ", current_blob_index)
                connections_from_current_blob = np.where((linksArray[:, :2] == [first_row[0], first_row[1]]).all(axis=1))[0]
                if(len(connections_from_current_blob) == 2):
                    # Time to split the track and not add the track, find out which are daughters and keep track
                    #print("Blob has daughter and split: ", first_row, connections_from_current_blob)
                    linksArray = np.delete(linksArray, (current_blob_index), axis = 0)
                    break
                next_blob_index = np.where((linksArray[:,:2] == [first_row[0] + 1, first_row[2]]).all(axis=1))[0] # Grab the index of the next blob from the array
                if next_blob_index.size == 0:
                    #print("Track ended break")
                    linksArray = np.delete(linksArray, (current_blob_index), axis = 0)
                    break
                else:
                    next_row = linksArray[next_blob_index[0]]
                    one_track[next_row[0]] = next_row[1]
                    linksArray = np.delete(linksArray, (current_blob_index), axis = 0)

                first_row = next_row
            #print(one_track)
            self.tracks.append(one_track)



class imgPropBundle(object):

    def __init__(self, imgFilename, imgStdSize=(96, 32)):
        self.imgFilename = imgFilename
        self.imgStdSize = imgStdSize
        self.img = imread(self.imgFilename)
        # props
        self.props = regionprops(label(self.img))
    
    def __len__(self):
        return len(self.props)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # accumulate properties you need for the image encoder net
        local_blob_prop = self.props[idx]
        # Use a torch tensor may be
        # bbox, centroid, area, length, ellipsicity
        bbox = list(local_blob_prop['bbox'])
        area = local_blob_prop['area']
        centroid = list(local_blob_prop['centroid'])
        eccentricity = local_blob_prop['eccentricity']
        major_axis_length = local_blob_prop['major_axis_length']
        minor_axis_length = local_blob_prop['minor_axis_length']
        perimeter = local_blob_prop['perimeter']
        image = local_blob_prop['image']

        # Do some operations on the image to center the image in the field,
        # to standardize the inputs to the network
        height, width = image.shape
        #print("Image shape before:", image.shape)
        # set blank background image

        # Generally wouldn't happen, just in-case safety measure
        # Should be redone in a different way than resizing
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
        

        # Construct properties list
        properties = []
        properties.extend(bbox)
        properties.append(area)
        properties.extend(centroid)
        properties.append(eccentricity)
        properties.append(major_axis_length)
        properties.append(minor_axis_length)
        properties.append(perimeter)

        return { 'props' : properties, 'image' : background}

    def plotImage(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        plt.figure()
        plt.imshow(self.__getitem__(idx)['image'])
        plt.show(block = False)

class oneSet(Dataset):
    """ This will  load a set of images for tracking training data
    One folder containes all the images of a channels.
    This class holds all the information together and helps in 
    loading on the GUI
    """

    def __init__(self, directory, transforms = None, fileformat = '.tiff'):

        self.directory = directory
        self.fileformat = fileformat
        self.transforms = transforms
        self.indices = [int(filename.split('/')[-1].split('.')[0]) for filename in glob.glob(self.directory + "*" + self.fileformat)]
        self.indices.sort()
        self.n_timepoints = len(glob.glob(self.directory + "*" + self.fileformat))

    def __len__(self):
        return self.n_timepoints

    def __str__(self):
        return 'directory: {self.directory} set'.format(self=self)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.directory + str(self.indices[idx]) + self.fileformat
        #print("Filename: ", filename)

        image = imread(filename, as_gray=True).astype('float32')
        label_image = label(image)
        regions = regionprops(label_image)
        bbox_list = []
        for prop in regions:
            bbox_list.append(prop.bbox)
        

        if self.transforms:
            image = self.transforms(image)

        return {'filename': filename, 'bbox': bbox_list, 'props': regions}
