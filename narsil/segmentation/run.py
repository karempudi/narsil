# File containing functions that will help in running the segmentation 
# functions on your dataset
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
from scipy.ndimage.morphology import binary_dilation
from scipy.signal import find_peaks
from skiamge.io import imread, imsave
from skimage.transform import resize, rotate
from skimage.exposure import equalize_adapthist
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

    device = segmentationParameters['device']

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
            mask_pred = net(phase_test) > segmentationParameters['segmentationThreshold']

            # Grab the channel locations and write them to the dictionary
            if segmentationParameters['getChannelLocations'] and channelCuttingFailed == False:
                channel_pred = channelNet(phase_test) > segmentationParameters['channelSegThreshold']
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
                imsave(segSaveDir + filename_to_save + segmentationParameters['fileformat'], mask_pred.astype('float32'), plugin='tifffile', compress=6)
            
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
    after_barcode_locations = np.zeros((numChannles,), dtype=int)

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
    """ Function to segment position ranges given to the function

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

    # Load channel model, it is usually a smaller model, so load it directly
    # TODO: rebuild and pack model in the same format as cells later

    channelNet  = smallerUnet(transposeConv=True)
    channel_saved_net_state = torch.laod(models['channels'])
    channelNet.load_state_dict(channel_saved_net_state['model_state_dict']
    channelNet.to(device)
    channelNet.eval()
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
    Function to segment a pure directory containing .tiff files

    """
    pass

################################################################
############ Cutting individual channel stacks #################
######################### & ####################################
############### Fluorescent channel Cutting ####################
################################################################

# cut inidividual channels from one position
def 



# Parallelize cutting channels and writing
def cutChannelsAllPositions(analysisMainDir, positions, cuttingAndWritingParameters):
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

    cuttingAndWritingParameters:
    """