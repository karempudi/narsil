# File containing functions that will help in running the segmentation 
# functions on your dataset
from scipy import ndimage
import torch
import os
import time
import numpy as np
import multiprocessing as mp
from narsil.segmentation.network import basicUnet, smallerUnet
from narsil.segmentation.datasets import phaseFolder
from narsil.utils.transforms import resizeOneImage, tensorizeOneImage
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes, binary_opening
from scipy.signal import find_peaks
from skimage.io import imread, imsave
from skimage.transform import resize, rotate
from skimage.exposure import equalize_adapthist
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import label, find_contours
from collections import OrderedDict
from functools import partial


############################################################
############################################################
############## Segmentation Functions ######################
######### (Cells and channel segmentaiton) #################
############################################################
############################################################

# Function to segment a directory conataining .tiff files
def segmentPosDirectory(positionPhaseDir, segCellNet, channelNet, segmentationParameters):
    """ Function to segment one directory containing images using the 
    parameters given in the segmentation parameters folder

    Arguments
    ---------
    positionPhaseDir: str
        Path of the directory containing phase contrast images that are to
        be segmented, string usually ends in position name ..../Pos101/ 
    segCellnet: torch model for cell segmentation
        Neural net loaded onto appropriate device and set to eval mode
	
	channelNet: torch model for channel segmentation
		Neural net loaded onto appropriate device and set to eval mode

    segmentationParameters: dict
        A dictionary containinig all the parameters needed for segmentation

    Returns
    -------
    Succcess/Failure: bool
        Success if everything goes well in this directory
    positionImgChop: dict
        A dictionary with keynames as imagefilenames and values are numpy arrays of the 
        locations where you can cut out the individual channels from the images
    """

    phaseImageTransform = segmentationParameters['transforms'] 

    device = torch.device(segmentationParameters['device'] if torch.cuda.is_available() else "cpu")

    positionNumber = int(positionPhaseDir.split('/')[-2][3:])
    #print(positionNumber)
    positionImgChopMap = {}

    if positionNumber in segmentationParameters['flipPositions']:
        flip = True
    else:
        flip = False
    
    phase_dataset = phaseFolder(positionPhaseDir, phaseDirName=segmentationParameters['phasePreset'],
                    transform=phaseImageTransform, phase_fileformat=segmentationParameters['fileformat'],
                    addNoise=segmentationParameters['addNoise'], flip=flip)

    phaseDataLoader = DataLoader(phase_dataset, segmentationParameters['batch_size'], shuffle=False, num_workers=6)

    channelCuttingFailed = False

    with torch.no_grad():
        for i_batch, data_test in enumerate(phaseDataLoader, 0):
            phase_test = data_test['phase'].to(device)
            filename_to_save = data_test['phase_filename'][0].split('.')[0].split('/')[-1]
            #print(filename_to_save)
            mask_pred = segCellNet(phase_test) 

            mask_pred = torch.sigmoid(mask_pred)

            # Grab the channel locations and write them to the dictionary
            if segmentationParameters['getChannelLocations'] and channelCuttingFailed == False:
                channel_pred = torch.sigmoid(channelNet(phase_test)) > segmentationParameters['channelSegThreshold']
                channel_pred = channel_pred.to("cpu").detach().numpy().squeeze(0).squeeze(0)
                #print(channel_pred.shape)
                try:
                    positionImgChopMap[filename_to_save] = getChannelLocationSingleImage(channel_pred, segmentationParameters['channelCuttingParameters'])
                
                except Exception as e:
                    channelCuttingFailed = True
                    print(e)
                    print(f"Pos{positionNumber} has failed cutting channels")
                    continue
            
            #print(f"{filename_to_save} ---> {channel_pred.shape}")
            # TODO: loop over if batch_size > 1
            mask_pred = mask_pred.to("cpu").detach().numpy().squeeze(0).squeeze(0)

            if segmentationParameters['useContourFinding']:
                mask_contour = np.zeros_like(mask_pred, dtype='bool')
                contours = find_contours(mask_pred, segmentationParameters['contoursThreshold'])
                for contour in contours:
                    mask_contour[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
                
                mask_contour = binary_fill_holes(mask_contour)
                mask_pred = mask_contour
            else:
                mask_pred = mask_pred > segmentationParameters['segmentationThreshold']

            if segmentationParameters['cleanSmallObjects']:
                mask_pred = remove_small_holes(mask_pred , area_threshold=30)
                mask_pred_labeled = label(mask_pred)
                mask_cleaned = remove_small_objects(mask_pred_labeled, min_size=40)
                mask_pred = mask_cleaned > 0

            if segmentationParameters['dilateAfterSeg']:
                mask_pred = binary_dilation(mask_pred)
            
            # clean up the regions and artifacts from the background
            if segmentationParameters['backgroundArtifactClean']:
                phase_original = data_test['phase_original'].numpy().squeeze(0).squeeze(0)
                #print(phase_original.shape)
                phase_original_thresholded1 = phase_original < segmentationParameters['minBackgroundOnPhase']
                phase_original_thresholded2 = phase_original > 0
                phase_original_thresholded = phase_original_thresholded1 * phase_original_thresholded2

                mask_pred = mask_pred * phase_original_thresholded

            if segmentationParameters['saveSeg']:
                segSaveDir = segmentationParameters['saveResultsDir'] + 'Pos' + str(positionNumber) + '/segmentedPhase/'
                if not os.path.exists(segSaveDir):
                    os.makedirs(segSaveDir)
                imsave(segSaveDir + filename_to_save + segmentationParameters['fileformat'], mask_pred.astype('float32'),
                    plugin='tifffile', compress=6, check_contrast=False)
            
            if segmentationParameters['savePhase']:
                phaseSaveDir = segmentationParameters['saveResultsDir'] + 'Pos' + str(positionNumber) + '/processedPhase/'
                phase_save = data_test['phase_original'].numpy().squeeze(0).squeeze(0)
                if not os.path.exists(phaseSaveDir):
                    os.makedirs(phaseSaveDir)
                imsave(phaseSaveDir + filename_to_save + segmentationParameters['fileformat'], phase_save.astype('float32'), plugin='tifffile', compress=6)
            
            if channelCuttingFailed == True:
                positionImgChopMap[filename_to_save] = np.asarray([])
        
        return (True, channelCuttingFailed, positionImgChopMap)

    # if doesn't return then something has failed during segmentation
    return (False, channelCuttingFailed, positionImgChopMap)


# Function to find channel locations to cut from the segmented channel image and some paramters
def getChannelLocationSingleImage(channel_pred, channelCuttingParameters):
    hist = np.sum(channel_pred[channelCuttingParameters['channel_min']: channelCuttingParameters['channel_max']], axis=0) > channelCuttingParameters['channel_sum']

    peaks, _ = find_peaks(hist, distance=channelCuttingParameters['histPeaksDistance'])

    indices_with_larger_gaps = np.where(np.ediff1d(peaks) > channelCuttingParameters['minBarcodeDistance'])[0]

    possible_barcode_locations = indices_with_larger_gaps[np.argmax(indices_with_larger_gaps > channelCuttingParameters['firstBarcodeIndex'])]

    #print(possible_barcode_locations)
    numChannels = channelCuttingParameters['numChannels']
    before_barcode_locations = np.zeros((numChannels,), dtype=int)
    after_barcode_locations = np.zeros((numChannels,), dtype=int)

    for i in range(numChannels):
        before_barcode_locations[i] = peaks[possible_barcode_locations-i]
        after_barcode_locations[i] = peaks[possible_barcode_locations+i+1]

    locations_to_cut = np.concatenate((before_barcode_locations[::-1], after_barcode_locations), axis = 0)

    return locations_to_cut



def loadNet(modelPath, device):
    """
    A function that takes in a model file and returns a net in eval mode 

    Arguments
    ---------
    modePath: .pth model file
        A model file, .pth file containing the model details to load the correct 
        model file

    device: str 
        usually "cuda:0" or "cuda:1"

    Returns
    -------
    net: torch.nn.Module object
        Net in eval mode
    """

    # Read the model file
    savedModel = torch.load(modelPath)

    # use the net depending on what model is loaded
    if savedModel['modelParameters']['netType'] == 'big':
        net = basicUnet(savedModel['modelParameters']['transposeConv'])
    elif savedModel['modelParameters']['netType'] == 'small':
        net = smallerUnet(savedModel['modelParameters']['transposeConv'])

    # load the net
    net.load_state_dict(savedModel['model_state_dict'])

    # send to device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    net.to(device)

    # eval mode
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    return net


def segmentAllPositions(phaseMainDir, positions, models, segmentationParameters):
    """ 
    Function to segment position ranges given to the function

    Arguments
    ---------
    phaseMainDir: str
        Path of the directory where positions with phase images live

    positions: range object or list 
        Range or list of positions in the directroy or the ones you want to 
        segment
    
    models: dict
        A dict with paths of different models (for segmentation of cells and 
            channels). "cells" and "channels" are keys and modelPaths are values

    segmentationParameters: dict
        Dict containing all the parameters needed for segmentation like GPU devices,
        image resize and shrink components, segmentation threshold, etc
    
    Returns
    -------
    None

    """
    start = time.time()

    device = segmentationParameters['device']

    net = loadNet(models['cells'], device)

    print("Segmentation Network loaded successfuly ...")


    channelNet  = loadNet(models['channels'], device)
    print("Channel Segmentation Network loaded successfully ... ")

    cuttingLocations = {}

    for position in positions:
        positionPhaseDir = phaseMainDir + 'Pos' + str(position) + '/'
        success, channelCuttingFailed, positionsImgChopMap = segmentPosDirectory(positionPhaseDir, net, channelNet, segmentationParameters)
        if (success and (channelCuttingFailed == False)):
            print(positionPhaseDir)
            cuttingLocations[position] = positionsImgChopMap

    duration = time.time() - start
    print(f"Duration for segmenting {positions} is {duration}s")

    print("Writing cutting locations ...")
    print(cuttingLocations.keys(), " positions have cutting locations done")
    np.save(segmentationParameters['saveResultsDir'] + 'channelLocations.npy', cuttingLocations)


def segmentDirectory(phaseDir, models, segmentationParameters, saveDir=None):
    """
    Function to segment a pure directory containing .tiff files. 
    You can use this to plot overlays as well.

    """
    return

################################################################
############ Cutting individual channel stacks #################
######################### & ####################################
############### Fluorescent channel Cutting ####################
################################################################

# cut inidividual channels from one position
def cutChannelsOnePosition(analysisPosDir, channelLocations, cuttingAndWritingParameters,
                 segmentedFileFormat='.tiff'):
    """
    Funciton to write and cut individual mother machine channels and write them for one
    position (which is a stack of images)
    """
    positionNumber = int(analysisPosDir.split('/')[-2][3:])

    # Directory where segmented images live, filenames are same as the raw data images
    segmentedPhaseDir = analysisPosDir + 'segmentedPhase/'
    
    blobsWriteDir = analysisPosDir + 'blobs/'
    channelWidth = cuttingAndWritingParameters['channelWidth']

    for i, filename in enumerate(channelLocations[positionNumber], 0):
        if (i > cuttingAndWritingParameters['cutUntilFrames']):
            return True
        
        imageFilename = segmentedPhaseDir + filename + segmentedFileFormat
        image = imread(imageFilename)

        peaks = channelLocations[positionNumber][filename]
        left = peaks - (channelWidth//2)
        right = peaks + (channelWidth//2)
        channelLimits = list(zip(left, right))

        for l in range(len(channelLimits)):
            if not os.path.exists(blobsWriteDir + str(l)):
                os.makedirs(blobsWriteDir + str(l))
            imgChop = image[:, channelLimits[l][0]: channelLimits[l][1]] * 255
            imsave(blobsWriteDir + str(l) + '/' + str(i) + segmentedFileFormat, imgChop, plugin='tifffile',
                         compress=6, check_contrast=False)
        
    return False



# Parallelize cutting channels and writing
def cutChannelsAllPositions(analysisMainDir, positions, cuttingAndWritingParameters,
        numProcesses=6):
    """
    Function to write and cut individual mother machine channels and write
    them to appropriate directories, for cell-tracking and growth rate analysis
    later

    Arguments
    ---------
    analysisMain: str
        Path of the analysis directory containing the segmented images directory

    positions: range object or list
        Range or list of position numbers to cut

    cuttingAndWritingParameters: dict
        Dictionary containing parameters used in channel cutting

    numProcesses: int
        Number of processes you want to parallelize on

    Returns
    -------
    None
    """
    start = time.time()

    channelLocationsFile = cuttingAndWritingParameters['saveResultsDir'] + 'channelLocations.npy'

    channelLocations = np.load(channelLocationsFile, allow_pickle=True).item()

    listPositionDirs = []
    for position in positions:
        if position not in channelLocations or len(channelLocations[position]) == 0:
            print(f"Skipping Pos{position} due to bad channel cutting ... ")
        else:
            listPositionDirs.append(analysisMainDir + 'Pos' + str(position) + '/')

    print(listPositionDirs)

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    
    pool = mp.Pool(processes=numProcesses)
    pool.map(partial(cutChannelsOnePosition, channelLocations=channelLocations, cuttingAndWritingParameters=cuttingAndWritingParameters), listPositionDirs)
    pool.close()
    pool.join()

    duration = time.time() - start
    print(f"Duration of cutting channels of {positions} is {duration}s")

    return None

###################################################################
################# Fluorescent cutting functions ###################
###################################################################

def cutFluorOnePosition(fluorPositionDir, channelLocations, fluorParameters):
    
    channelNames = fluorParameters['channelNames']
    positionNumber = int(fluorPositionDir.split('/')[-2][3:])
    channelWidth = fluorParameters['channelWidth']
    fluorTransform = fluorParameters['transform'] 

    writeDirectory = fluorParameters['saveResultsDir'] + 'Pos' + str(positionNumber) + '/fishChannels/'

    for channelName in channelNames:

        channelImageFileName = fluorPositionDir + channelName + '/' + fluorParameters['fluorImageName'] + '.tiff'
        channelImage = imread(channelImageFileName, as_gray=True)

        height, width = channelImage.shape
        if positionNumber in fluorParameters['flipPositions']:
            channelImage = rotate(channelImage, angle=180, preserve_range=True)

        # Here you can apply the transformations need to clean up the image
        channelImage = fluorTransform(channelImage)
        channelImage = channelImage.astype('uint16')
        if fluorParameters['equalize'] == 'equalize_adapthist':
            channelImage = (65535 * equalize_adapthist(channelImage))


        peaks = channelLocations[positionNumber][fluorParameters['phaseImageToMap']]
        left = peaks - (channelWidth//2)
        right = peaks + (channelWidth//2)
        channelLimits = list(zip(left, right))

        for l in range(len(channelLimits)):
            if not os.path.exists(writeDirectory + str(l)):
                os.makedirs(writeDirectory + str(l))
            imsave(writeDirectory + str(l) + '/' + channelName + '.tiff', channelImage[:, channelLimits[l][0] : channelLimits[l][1]], compress=6)

    print(f"FluorCutting in Pos{positionNumber} Done ..")
    return None

    

def cutFluorAllPositions(fluorMainDir, positions, fluorParameters, numProcesses=6):

    start = time.time()
    channelLocationsFilename = fluorParameters['saveResultsDir'] + 'channelLocations.npy'
    print(f"Reading {channelLocationsFilename} to get locations to cut ... ")

    channelLocations = np.load(channelLocationsFilename, allow_pickle=True).item()

    listPositionDirs = []
    for position in positions:
        if (position not in channelLocations) or len(channelLocations[position]) == 0:
            print(f"Skipping Pos{position} flour cutting due to bad channel detection ...")
        else:
            listPositionDirs.append(fluorMainDir + 'Pos' + str(position) + '/')
    print(listPositionDirs)

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    pool = mp.Pool(processes=numProcesses)
    pool.map(partial(cutFluorOnePosition, channelLocations=channelLocations, fluorParameters=fluorParameters), listPositionDirs)
    pool.close()
    pool.join()

    duration = time.time() - start
    print(f"Duration of cutting Fluorescent channels of {positions} is {duration}s")

    return None
 

