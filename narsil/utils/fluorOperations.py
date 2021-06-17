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
from skimage.restoration import rolling_ball
from skimage.filters import difference_of_gaussians, gaussian
from skimage.exposure import equalize_hist
from skimage.morphology import area_opening
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.util.dtype import img_as_ubyte
from skimage import img_as_ubyte

# Function to generate weight maps from a binary segmentation file. 
# The binary segmenation where dtype is uint8 with true values == 255
def generateWeights(filename, sigma = 5, w0 = 10):
	img = imread(filename)
	# removing objects and calculating distances to objects needs labelled images
	labeledImg, num = label(img, return_num=True, connectivity=2)
	# remove small objects
	labeledImg = remove_small_objects(labeledImg, min_size=250)
	# unique values == number of blobs
	unique_values = np.unique(labeledImg) 
	num_values = len(unique_values)
	h, w = labeledImg.shape
	# stack keeps distance maps each blob
	stack = np.zeros(shape=(num_values, h, w))
	for i in range(num_values):
		stack[i] = ndi.distance_transform_edt(~(labeledImg == unique_values[i]))
	# sort the distance
	sorted_distance_stack = np.sort(stack, axis=0)
	# d1 and d2 are the shortest and second shortest distances to each object, 
	# sorted_distance_stack[0] is distance to the background. One can ignore it
	distance_sum = sorted_distance_stack[1] + sorted_distance_stack[2]
	squared_distance = distance_sum ** 2/ (2 * (sigma**2))
	weightmap = w0 * np.exp(-squared_distance)*(labeledImg == 0)
	return weightmap

# Used for generating binary fluorescent mask from fluorescence data,
# Tested against Pseudomonas data only
def generateBinaryMaskFromFluor(fluorImgfilename):
	fluorImg = imread(fluorImgfilename).astype('float32')
	background = rolling_ball(fluorImg, radius = 50)
	fluorImg_remback = fluorImg - background
	filtered = difference_of_gaussians(fluorImg_remback, 1, 4)
	fluorImg_scaled_eq = equalize_hist(filtered)
	filtered_gaussian = gaussian(fluorImg_scaled_eq)
	filtered_median = ndi.median_filter(filtered_gaussian, size=5)
	filtered_gaussian  = filtered_gaussian > 0.92
	image_opening = area_opening(filtered_gaussian, area_threshold=150)
	distance = ndi.distance_transform_edt(image_opening)
	coords = peak_local_max(distance,min_distance = 90, footprint=np.ones((7,7)), labels=image_opening)
	mask = np.zeros(distance.shape, dtype=bool)
	mask[tuple(coords.T)] = True
	markers, _ = ndi.label(mask)
	labels = watershed(-distance, markers, mask=image_opening, watershed_line=True, connectivity=2)
	return img_as_ubyte(labels > 0)