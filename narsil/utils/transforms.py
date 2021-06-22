from matplotlib.pyplot import fill
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import torchvision.transforms.functional as TF
from skimage import transform

class padTo16(object):

    def __call__(self, sample):
        
        width, height = sample['phase'].size
        pad_width = 16 - (width % 16) 
        pad_height = 16 - (height % 16)

        sample['phase'] = TF.pad(sample['phase'], padding=[0, 0, pad_width, pad_height], padding_mode="constant", fill=0)
        sample['mask'] = TF.pad(sample['mask'], padding=[0, 0, pad_width, pad_height], padding_mode="constant", fill=0)
        sample['weights'] = TF.pad(sample['weights'], padding=[0, 0, pad_width, pad_height], padding_mode="constant", fill=0)

        return sample


class changedtoPIL(object):

    def __call__(self, sample):
        phaseImg = sample['phase'].astype('int32')
        maskImg = sample['mask']
        weightsImg = sample['weights'].astype('float32')

        sample['phase'] = TF.to_pil_image(phaseImg)
        sample['mask'] = TF.to_pil_image(maskImg)
        sample['weights'] = TF.to_pil_image(weightsImg)
        
        return sample

class randomCrop(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        
        i, j, h, w = transforms.RandomCrop.get_params(sample['phase'], output_size=(self.output_size, self.output_size))

        sample['phase'] = TF.crop(sample['phase'], i, j, h, w)
        sample['mask'] = TF.crop(sample['mask'], i, j, h, w)
        sample['weights'] = TF.crop(sample['weights'], i, j, h, w)
        return sample

class randomRotation(object):

    def __init__(self, rotation_angle):
        self.rotation_angle = rotation_angle

    def __call__(self, sample):

        angle = transforms.RandomRotation.get_params(self.rotation_angle)
        sample['phase'] = TF.rotate(sample['phase'], angle)
        sample['mask'] = TF.rotate(sample['mask'], angle)
        sample['weights'] = TF.rotate(sample['weights'], angle)

        return sample

class randomAffine(object):

    def __init__(self, scale, shear):
        self.scale = scale # something like 0.75-1.25
        self.shear = shear # something like [-30, 30, -30, 30]

    def __call__(self, sample):

        angle, translations, scale, shear = transforms.RandomAffine.get_params(degrees=[0, 0], translate=None, 
                scale_ranges=self.scale, shears=self.shear, img_size=sample['phase'].size)

        sample['phase'] = TF.affine(sample['phase'], angle=angle, translate=translations, scale=scale, shear=shear)
        sample['mask'] = TF.affine(sample['mask'], angle=angle, translate=translations, scale=scale, shear=shear)
        sample['weights'] = TF.affine(sample['weights'], angle=angle, translate=translations, scale=scale, shear=shear)
        return sample

class randomBrightness(object):

    def __init__(self, probability):
        self.probability = probability

    def __call__(self, sample):
        
        if random.random() < self.probability:

            if random.random() < 0.5:
                brightness_factor = 1.0 + random.random()/5.0 # random 20% adjustment
                sample['phase'] = TF.adjust_brightness(sample['phase'], brightness_factor=brightness_factor)
            else:
                brightness_factor = 1.0 - random.random()/5.0 # random 20% adjustment
                sample['phase'] = TF.adjust_brightness(sample['phase'], brightness_factor=brightness_factor)

        return sample

class randomContrast(object):

    def __init__(self, contrast_factor, probability):
        self.contrast_factor = contrast_factor
        self.probability = probability
    
    def __call__(self, sample):
        
        if random.random() < self.probability:
            sample['phase'] = TF.adjust_contrast(sample['phase'], self.contrast_factor)

class toTensor(object):

    def __call__(self, sample):
        sample['phase'] = transforms.ToTensor()(np.array(sample['phase']).astype(np.float32))
        sample['mask'] = transforms.ToTensor()(np.array(sample['mask']).astype(np.float32))
        sample['weights'] = transforms.ToTensor()(np.array(sample['weights']).astype(np.float32))

        sample['phaseFilename'] = str(sample['phaseFilename'])
        sample['maskFilename'] = str(sample['maskFilename'])
        sample['weightsFilename'] = str(sample['weightsFilename'])
        return sample

class normalize(object):

    def __call__(self, sample):

        sample['phase'] = (sample['phase'] - torch.mean(sample['phase']))/torch.std(sample['phase'])

        # mask are 255 for True and 0 for false
        sample['mask'] = sample['mask']/255.0

        #sample['weights'] += 1.0
        

        return sample

class addnoise(object):

    def __init__(self, mean=0.0, std=0.15):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        sample['phase'] = sample['phase'] + torch.randn(sample['phase'].size()) * self.std + self.mean

        return sample


class resizeOneImage(object):

    def __init__(self, imgResizeShape, imgToNetSize):
        assert isinstance(imgResizeShape, tuple)
        assert isinstance(imgToNetSize, tuple)
        self.imgResizeShape = imgResizeShape
        self.imgToNetSize = imgToNetSize

    def __call__(self, image):

        height, width = image.shape
        # check if imageSize is bigger or equal to the 
        image = np.pad(image, pad_width=((0, self.imgResizeShape[0] - height), (0, self.imgResizeShape[1] - width)), 
                          mode='constant', constant_values = 0.0)
        if self.imgResizeShape[0] != self.imgToNetSize[0] or self.imgResizeShape[1] != self.imgToNetSize[1]:
            # Net size is not same a resized image
            image = transform.resize(image, self.imgToNetSize, anti_aliasing=True, preserve_range=True)
        return image


class tensorizeOneImage(object):
    def __init__(self, numUnsqueezes=1):
        self.numUnsqueezes = numUnsqueezes
    def __call__(self, phase_image):
        phase_image = phase_image.astype('float32')
        if self.numUnsqueezes == 1:
            return torch.from_numpy(phase_image).unsqueeze(0)
        elif self.numUnsqueezes == 2:
            return torch.from_numpy(phase_image).unsqueeze(0).unsqueeze(0)


class stripAxis(object):

    def __call__(self, imageTensor):
        return imageTensor.to("cpu").detach().numpy().squeeze(0).squeeze(0)

