import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import torchvision.transforms.functional as TF

class padTo16(object):

    def __call__(self, sample):
        pass


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

        sample['weights'] += 1.0
        

        return sample

class addnoise(object):

    def __init__(self, mean=0.0, std=0.15):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        sample['phase'] = sample['phase'] + torch.randn(sample['phase'].size()) * self.std + self.mean

        return sample

class WeightedUnetLoss(nn.Module):
    """
    Custom loss function, for Unet, BCE + DICE + weighting
    """
    def __init__(self):
        super(WeightedUnetLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, output, target, weights):

        #output_weighted = torch.sigmoid(output) 

#        output_weighted = output
#        output_weighted = torch.sigmoid(output_weighted)

#        output_flat = output_weighted.view(-1)
#        target_flat = target.view(-1)


        #bce_loss = self.bce_loss(output_flat, target_flat)
        #bce_loss = F.binary_cross_entropy_with_logits(output, target, weight = weights)
#        bce_loss = F.binary_cross_entropy(output_flat, target_flat)
#
#        batch_size = target.shape[0]
#
#        output = torch.sigmoid(output)
#        output_dice = output.view(batch_size, -1)
#        target_dice = target.view(batch_size, -1)
#
#        intersection = (output_dice * target_dice)
#        dice_per_image = 2. * (intersection.sum(1)) / (output_dice.sum(1) + target_dice.sum(1))
#
#        dice_batch_loss = 1 - dice_per_image.sum() / batch_size
#        print(dice_batch_loss.item(), bce_loss.item())
        #return 0.5 * bce_loss + dice_batch_loss

        output = output * weights

        output = torch.sigmoid(output)

        batch_size = target.shape[0]


        output_reshaped = output.view(batch_size, -1)
        target_reshaped = target.view(batch_size, -1)

        bce_loss = F.binary_cross_entropy(output_reshaped, target_reshaped)
        intersection = (output_reshaped * target_reshaped)
        dice_per_image = 2. * (intersection.sum(1)) / (output_reshaped.sum(1) + target_reshaped.sum(1))

        dice_batch_loss = 1 - dice_per_image.sum() / batch_size
        print(dice_batch_loss.item(), bce_loss.item())
        return 0.5 * bce_loss + 2.5 * dice_batch_loss
       #return bce_loss

