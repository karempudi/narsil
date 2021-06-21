# File containing functions that will help in running the segmentation 
# functions on your dataset
import torch
import os
import time
import numpy as np
import multiprocessing as mp
from narsil.segmentation.network import basicUnet, smallerUnet
from narsil.segmentation.datasets import phaseFolder, 
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

# Function to segment a directory conataining .tiff files
def segmentDirectory(positionPhaseDir, segCellNet, channelNet, segmentationParameters):
    """ Function to segment one directory containing images using the 
    parameters given in the segmentation parameters folder

    Arguments
    ---------
    positionPhaseDir: str
        Path of the directory containing phase contrast images that are to
        be segmented
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

	
