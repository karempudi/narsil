# File containing datasets used for training the siamese net
import glob
from skimage.io import imread, imsave
import numpy as np

class channelDataset(object):

    def __init__(self, directory, frame_skip=1, fileformat='.tiff'):

        self.directory = directory
        self.fileformat = fileformat
        self.n_images = len(glob.glob(self.directory + "*" + self.fileformat))
        self.indices = [int(filename.split('.')[0].split('/')[-1]) for filename in 
                        glob.glob(self.directory + "*" + self.fileformat)]
        self.indices.sort()
        self.indices = self.indices[::frame_skip]

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):

        channel_img_name =  self.directory + str(self.indices[idx]) + self.fileformat
        return imread(channel_img_name)


