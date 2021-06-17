# File containing the operations that are done on fluorescent Images
# to obtain binary segmentation mask and weight masks
import numpy as np
from scipy import ndimage as ndi
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import glob
import pathlib

# Function to generate weight maps from a binary segmentation file
def generateWeights(filename):
	pass

#
def generateBinaryMaskFromFluor():
	pass