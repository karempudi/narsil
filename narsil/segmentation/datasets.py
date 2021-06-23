import glob
import os
import random
from skimage import io
from skimage.exposure.exposure import adjust_gamma, equalize_hist
from skimage.exposure.histogram_matching import match_histograms
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pathlib
from narsil.utils.transforms import resizeOneImage, tensorizeOneImage, stripAxis
from torchvision import transforms
from skimage.transform import resize, rotate


class mmDataset(Dataset):
    """ mm Segmentation Mask dataset for Training/Testing/Validation"""

    def __init__(self, phase_dir, mask_dir, scaleFactor=None,transform = None,
             phase_fileformat='.tiff', mask_fileformat = '.tiff', add_noise = False,
             augmentedIllumination=False):
        """
        Args:
              phase_dir (string): Name of the phase channel directory
              mask_dir (string): Name of the mask directory (use ImageJ)
              phase_transform : A series of transforms you want to apply to the images
                                (usually a crop and rotate if needed)
              mask_transform: A series of transforms that is needed to generate a segmask
                                (or just some crops, same as phase, if they are already
                                 preprocessed in ImageJ)   
              fileformat: Give the fileformat like .tiff, or .png for 
        """
        # read the folder and count number of images and set the transform
        self.phase_fileformat = phase_fileformat
        self.mask_fileformat = mask_fileformat
        self.n_images = len(glob.glob(phase_dir + "*" + phase_fileformat))
        self.phase_dir = phase_dir
        self.mask_dir = mask_dir
        self.indices = [filename.split('.')[0].split('/')[-1] for filename in glob.glob(phase_dir + "*" + phase_fileformat)] 
        self.transform = transform
        self.add_noise = add_noise
        self.scaleFactor = scaleFactor
        self.augmentedIllumination = augmentedIllumination

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        phase_img_name = self.phase_dir + self.indices[idx] + self.phase_fileformat
        #mask_img_name = self.mask_dir + self.indices[idx] + '_mask' + self.mask_fileformat
        mask_img_name = self.mask_dir + self.indices[idx] + self.mask_fileformat
        
        phase_img = io.imread(phase_img_name, as_gray=True)
        #if random.random() > 0.25:
        #phase_img = adjust_gamma(phase_img, gamma=1.2)

        phase_img = phase_img.astype('float32')
        #phase_img = equalize_hist(phase_img)
        mask_img = io.imread(mask_img_name, as_gray=True).astype('bool')

        #height, width = phase_img.shape
        #if self.augmentedIllumination and random.random() > 0.5:
        #    phase_img = phase_img + np.array([np.linspace(1000, 8000, num=width),]*height)

        #phase_original = np.copy(phase_img)
        #mask_original = np.copy(mask_img)

        #phase_mean  = np.mean(phase_img)
        #phase_img = phase_img
        #phase_std = np.std(phase_img)

        #phase_img = (phase_img - phase_mean) / phase_std
        
        # zero everything that is above 2 standard deviations
        
        #indices = np.where(phase_img > 2)
        #mask_img[indices[0], indices[1]] = 0
        #if self.add_noise:
        #    rand_num = np.random.normal(0,0.15, phase_img.shape)
        #    phase_img = phase_img + rand_num

        sample = {'phase': phase_img, 'mask': mask_img , 'phase_filename': phase_img_name, 'mask_filename': mask_img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample

class phaseFolder(Dataset):
    """
    Binding class that brings phase images in a directory together for batching and segmenting

    main Directory is going to be a positon directory and not the main main Dir containing positions
    """

    def __init__(self, mainDirectory, phaseDirName = "phase/", transform = None, phase_fileformat = '.tiff', addNoise = False, flip = False):
        self.mainDirectory = mainDirectory
        self.phase_fileformat = phase_fileformat
        self.transform = transform
        self.indices = [filename.split('.')[0].split('/')[-1] for filename in glob.glob(mainDirectory + phaseDirName +  "*" + phase_fileformat)]
        self.indices.sort()
        self.n_images = len(self.indices)
        self.phaseDirName = phaseDirName
        self.addNoise = addNoise
        self.flip = flip



    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        phase_img_name = self.mainDirectory + self.phaseDirName + self.indices[idx] + self.phase_fileformat

        phase_img = io.imread(phase_img_name, as_gray = True).astype('float32')
        #phase_img = match_histograms(phase_img, self.img_ref).astype('float32')

        phase_mean = np.mean(phase_img)
        phase_std = np.std(phase_img)

        phase_img_normalized = (phase_img - phase_mean) / phase_std


        if self.addNoise:
            rand_num = np.random.normal(0, 0.15, phase_img.shape)
            phase_img_normalized = phase_img_normalized + rand_num

        if self.flip:
            phase_img_normalized = rotate(phase_img_normalized, angle = 180)
            phase_img = rotate(phase_img, angle=180)
        

        if self.transform:
            phase_img_normalized = self.transform(phase_img_normalized)
            phase_img = self.transform(phase_img)

        return {'phase' : phase_img_normalized, 'phase_filename' : phase_img_name, 'phase_original': phase_img}

class phaseTestDir(object):
    """
    Binding class that brings phase images in a directory together for batching and segmenting

    main Directory is going to be a positon directory and not the main main Dir containing positions
    """
    def __init__(self, phaseDirectory, transform = None, phase_fileformat = '.tiff', addNoise = False, flip = False):
        self.phaseDirectory = phaseDirectory
        self.phase_fileformat = phase_fileformat
        self.transform = transform
        self.indices = [filename.split('.')[0].split('/')[-1] for filename in glob.glob(phaseDirectory + "*" + phase_fileformat)]
        self.n_images = len(self.indices)
        self.addNoise = addNoise
        self.flip = flip


    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        phase_img_name = self.phaseDirectory + self.indices[idx] + self.phase_fileformat

        phase_img = io.imread(phase_img_name, as_gray = True)
        #phase_img = adjust_gamma(phase_img, gamma=1.2)
        phase_img = phase_img.astype('float32')

        


        phase_mean = np.mean(phase_img)
        phase_std = np.std(phase_img)

        phase_img_normalized = (phase_img - phase_mean) / phase_std

        #phase_img[phase_img > 1.3] += 0.5

        #phase_img = match_histograms(phase_img, self.imgref)
        #phase_img = equalize_hist(phase_img)

        #phase_img[phase_img > 1.0] += 0.5
        

        if self.addNoise:
            rand_num = np.random.normal(0, 0.15, phase_img_normalized.shape)
            phase_img_normalized = phase_img_normalized + rand_num

        if self.flip:
            phase_img_normalized = rotate(phase_img_normalized, angle = 180)
        

        if self.transform:
            phase_img_normalized = self.transform(phase_img_normalized)

        return phase_img_normalized


"""
Class for clikcing through the segmentation and phase images results of any directory
using a GUI to verify quality of the network visually.
"""
class segViewer(object):

    def __init__(self, phaseDir, net, threshold, addNoise = False, shrinkInHalf = False, device = "cpu",fileformat='.tiff'):
        self.phaseDir = phaseDir
        self.fileformat = fileformat
        self.threshold = threshold
        self.net = net # net should already be loaded and in eval state
        self.transforms = None
        self.shrinkInHalf = shrinkInHalf
        self.addNoise = addNoise
        self.strip = stripAxis()
        self.device = torch.device(device)
        self.filenames = glob.glob(self.phaseDir + "*" + self.fileformat)

        self.fig, self.ax = plt.subplots(1, 1, num=str(self.phaseDir))
        self.axcolor = 'lightgoldenrodyellow'
        plt.subplots_adjust(left = 0.25,bottom=0.25)


        # set the currnetPhase Image and segImage
        self.currentImageIndex = 0
        self.currentPhaseImage = self.__getitem__(self.currentImageIndex) # will be a tensor
        with torch.no_grad():
            self.currentSegImage = self.net(self.currentPhaseImage) > self.threshold

        self.pltImage = self.ax.imshow(self.strip(self.currentPhaseImage))

        # buttons
        self.nextax = plt.axes([0.65, 0.025, 0.1, 0.03])
        self.previousax = plt.axes([0.8, 0.025, 0.1, 0.03])
        self.nextButton = Button(self.nextax, 'Next', color=self.axcolor, hovercolor='0.975')
        self.previousButton = Button(self.previousax, 'Previous', color=self.axcolor, hovercolor='0.975')
        self.nextButton.on_clicked(self.next)
        self.previousButton.on_clicked(self.previous)

        # toggle between phase and seg
        self.toggleNames = ('Phase', 'Segmentation')
        self.togglesax = plt.axes([0.025, 0.5, 0.15, 0.15])
        self.toggles = RadioButtons(self.togglesax, self.toggleNames)
        self.toggles.on_clicked(self.updateImage)

        plt.pause(0.1)
    
    # this will give out the tensor needed to go directly into the net
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        phaseFileName = self.filenames[idx]
        phaseImage = io.imread(phaseFileName).astype('float32')
        
        (height, width) = phaseImage.shape
        if height%32 != 0:
            reshapeHeight = (height//32 + 1) * 32
        else:
            reshapeHeight = height
        
        if width%32 != 0:
            reshapeWidth = (width//32 + 1) * 32
        else:
            reshapeWidth = width

        resizeShape = (reshapeHeight, reshapeWidth)

        phase_mean = np.mean(phaseImage)
        phase_std = np.std(phaseImage)
        phaseImage = (phaseImage - phase_mean) / phase_std

        if self.addNoise:
            rand_num = np.random.normal(0, 0.15, phaseImage.shape)
            phaseImage = phaseImage + rand_num


        if self.shrinkInHalf:
            shrinkedHeight = reshapeHeight//2
            shrinkedWidth = reshapeWidth//2
            shrinkShape = (shrinkedHeight, shrinkedWidth)
            self.transforms = transforms.Compose([resizeOneImage(resizeShape,shrinkShape), tensorizeOneImage(2)])
        else:
            self.transforms = transforms.Compose([resizeOneImage(resizeShape, resizeShape),tensorizeOneImage(2)])

        if self.transforms != None:
            phaseImage = self.transforms(phaseImage)
        
        print(phaseImage.shape)

        return phaseImage.to(self.device)


    def __len__(self):
        return len(self.filenames)

    def next(self, val):
        if self.currentImageIndex != self.__len__() - 1:
            self.currentImageIndex += 1
        
        self.currentPhaseImage = self.__getitem__(self.currentImageIndex)
        with torch.no_grad():
            self.currentSegImage = self.net(self.currentPhaseImage) > self.threshold

        self.redraw()    

    def redraw(self):
        if self.toggles.value_selected == 'Phase':
            print("Phase is active")
            self.pltImage.set_data(self.strip(self.currentPhaseImage))
            self.fig.canvas.draw()
        elif self.toggles.value_selected == 'Segmentation':
            self.pltImage.set_data(self.strip(self.currentSegImage))
            self.fig.canvas.draw()


    def previous(self, val):
        if self.currentImageIndex != 0:
            self.currentImageIndex -= 1

        self.currentPhaseImage = self.__getitem__(self.currentImageIndex)
        with torch.no_grad():
            self.currentSegImage = self.net(self.currentPhaseImage) > self.threshold

        self.redraw()

    def updateImage(self, label):
        if label == 'Segmentation':
            self.pltImage.set_data(self.strip(self.currentSegImage))
            print("Segmenation",self.currentSegImage.shape)
            self.fig.canvas.draw()
        elif label == 'Phase':
            self.pltImage.set_data(self.strip(self.currentPhaseImage))
            print("Phase", self.currentPhaseImage.shape)
            self.fig.canvas.draw()





class fishdataFolder(object):
    """
    Binding class that brings information about 
    a particular channel of the flourescent channels
    Initialization:
    Parameters:
        1. fishMainDirectory (string) ---> Main directory where the Positions Folders are
        2. channel names (list of strings): Ex --> 488, cy5, cy3, etc 
        3. species -> channel map (dictionary) : Ex --> {'ecoli':[488, cy5], 'klebsiella': [cy3], 'cocci': [488]}
        4. flipPositions -> range Object or a list of position that need to be flipped
    """

    def __init__(self, fishPositionDir, channelNames, speciesChannelMap, flipPositions=None):

        self.fishPositionDir = fishPositionDir
        self.channelNames = channelNames
        self.speciesChannelMap = speciesChannelMap
        self.flipPositions = ['Pos' + str(i) for i in flipPositions]
        self.positionNumber = int(self.fishPositionDir.split('/')[-2][3:])
        # Grab the position names for the main directory
        #self.positions = [positionName for positionName in os.listdir(self.mainDirectory) if os.path.isdir(os.path.join(self.mainDirectory, positionName))]
        # sort positions based on its numbers
        #self.positions = sorted(self.positions, key= lambda position: int(position[3:]))

    def __len__(self):
        return len(self.channelNames)

    def __getitem__(self, idx):
        
        channelDirectories = {}
        for channelName in self.channelNames:
            channelDirectories[channelName] = self.mainDirectory + position + '/' + channelName + '/'

        channelDirectories['position'] = position
        if position in self.flipPositions:
            channelDirectories['flip'] = True
        else:
            channelDirectories['flip'] = False
        return channelDirectories


    def printSpeciesProbeMap(self):
        return None


class mmDataMultiSpecies(object):

    def __init__(self, trainingDataDir, species, transforms = None, datasetType='train', includeWeights=False):
        self.trainingDataDir = pathlib.Path(trainingDataDir)
        self.species = species
        self.datasetType = datasetType
        self.transforms = transforms
        self.includeWeights = includeWeights
        # construct all the file names  for all the species in the images, species dir will also have an
        # empty dir, containing empty channels, don't worry about this
        self.phaseDirs = []
        self.maskDirs = []
        self.weightDirs = []
        for dirName in self.species:
            self.phaseDirs.append(self.trainingDataDir.joinpath(dirName , 'phase_' + self.datasetType))
            self.maskDirs.append(self.trainingDataDir.joinpath(dirName, 'mask_' + self.datasetType))
            if self.includeWeights:
                self.weightDirs.append(self.trainingDataDir.joinpath(dirName, 'weights_' + self.datasetType))

        self.phaseFilenames = []
        self.maskFilenames = []
        self.weightFilenames = []

        for i, directory in enumerate(self.phaseDirs, 0):
            # grab filenames and add the corresponding mask to the file list
            filenames = [filename.name for filename in directory.glob('*.tif')]  
            for filename in filenames:
                # adding phaseFilenames
                self.phaseFilenames.append(directory.joinpath(filename))
                # adding maskFilenames
                self.maskFilenames.append(self.maskDirs[i].joinpath(filename))
                # adding weightFilenames
                if self.includeWeights:
                    self.weightFilenames.append(self.weightDirs[i].joinpath(filename))

    def __len__(self):
        return len(self.phaseFilenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        phaseImg = io.imread(self.phaseFilenames[idx])
        maskImg = io.imread(self.maskFilenames[idx])

        if self.includeWeights:
            weightImg = io.imread(self.weightFilenames[idx])
            weightFilename = self.weightFilenames[idx]
        else:
            weightImg = np.zeros(phaseImg.shape)
            weightFilename = None

        sample = {'phase': phaseImg, 'mask': maskImg, 'weights': weightImg,
                'phaseFilename': self.phaseFilenames[idx], 'maskFilename': self.maskFilenames[idx],
                'weightsFilename': weightFilename}
        
        if self.transforms != None:
            sample = self.transforms(sample)

        return sample

    # This will plot the data point that goes into the training net
    def plotDataPoint(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        if type(idx) == list:
            print("Plotter only works with integer indices, can only plot one item at a time :( ")
            return
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
            data = self.__getitem__(idx)
            ax1.imshow(data['phase'].numpy().squeeze(0), cmap='gray')
            #ax1.set_xlabel(str(data['phaseFilename']))
            ax1.set_xlabel('Phase')
            #ax1.set_title('Phase')

            ax2.imshow(data['mask'].numpy().squeeze(0), cmap='gray')
            #ax2.set_xlabel(str(data['maskFilename']))
            ax2.set_xlabel('Binary mask')
            #ax2.set_title('mask')

            ax3.imshow(data['weights'].numpy().squeeze(0))
            #ax3.set_xlabel(str(data['weightFilename']))
            ax3.set_xlabel('Weight map')
            #ax3.set_title('weights')
            plt.show()



