# File containing various loss functions
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedUnetLoss(nn.Module):
	"""
	Custom loss function for Unet BCE + DICE + weighting
	"""

	def __init__(self):
		super(WeightedUnetLoss, self).__init__()

	def forward(self, output, target, weights):

		output = torch.sigmoid(output)

		batch_size = target.shape[0]


		target_weighted = target - weights
	
		output_reshaped = output.view(batch_size, -1)
		target_weighted_reshaped = target_weighted.view(batch_size, -1)



		bce_loss = F.binary_cross_entropy(output_reshaped, target_weighted_reshaped)

		target_reshaped = target.view(batch_size, -1)

		intersection = (output_reshaped * target_reshaped)
		dice_per_image = 2. * (intersection.sum(1)) / (output_reshaped.sum(1) + target_reshaped.sum(1))
		dice_batch_loss = 1 - dice_per_image.sum() / batch_size
		print(dice_batch_loss.item(), bce_loss.item())
		return 0.5 * bce_loss  + 2.0 *dice_batch_loss
	#return bce_loss


class WeightedUnetLossExact(nn.Module):
	"""
	Custom loss function that implements the exact formulation of weighted BCE in 
	from the U-net paper https://arxiv.org/pdf/1505.04597.pdf
	"""

	def __init__(self):
		super(WeightedUnetLossExact, self).__init__()

	def forward(self, output, target, weights):
		
		output = torch.sigmoid(output)

		batch_size = target.shape[0]

		# calculate intersection over union
		target_reshaped = target.view(batch_size, -1)
		output_reshaped = output.view(batch_size, -1)
		intersection = (output_reshaped * target_reshaped)
		dice_per_image = 2. * (intersection.sum(1)) / (output_reshaped.sum(1) + target_reshaped.sum(1))
		dice_batch_loss = 1 - dice_per_image.sum() / batch_size

		# weighted cross entropy function

		weights_reshaped = weights.view(batch_size, -1)  + 1.0
		bce_loss = weights_reshaped * F.binary_cross_entropy(output_reshaped, target_reshaped, reduction='none')

		weighted_entropy_loss = bce_loss.mean()		
		print(dice_batch_loss.item(), weighted_entropy_loss.item())
		return dice_batch_loss  + 0.5 * weighted_entropy_loss



class UnetLoss(nn.Module):
	"""
	Custom loss function, for Unet, BCE + DICE
	"""
	def __init__(self):
		super(UnetLoss, self).__init__()
		self.bce_loss = nn.BCELoss()

	def forward(self, output, target):

		output = torch.sigmoid(output)
		output_flat = output.view(-1)
		target_flat = target.view(-1)

		bce_loss = self.bce_loss(output_flat, target_flat)

		batch_size = target.shape[0]

		output_dice = output.view(batch_size, -1)
		target_dice = target.view(batch_size, -1)

		intersection = (output_dice * target_dice)
		dice_per_image = 2. * (intersection.sum(1)) / (output_dice.sum(1) + target_dice.sum(1))

		dice_batch_loss = 1 - dice_per_image.sum() / batch_size
		print(dice_batch_loss.item(), bce_loss.item())
		return 0.5 * bce_loss + dice_batch_loss
		#return bce_loss

class BCELoss(nn.Module):

	def __init__(self):
		super(BCELoss, self).__init__()
		self.bce_loss = nn.BCELoss()

	def forward(self, output, target):

		output = torch.sigmoid(output)

		output_flat = output.view(-1)
		target_flat = target.view(-1)

		bce_loss = self.bce_loss(output_flat, target_flat)

		return bce_loss


class ContrastiveLoss(nn.Module):
	"""
	Contrastive Loss function
	"""
	def __init__(self, margin=10.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, output1, output2, label):
		euclidian_distance = F.pairwise_distance(output1, output2, keepdim=True)
		contrastive_loss = (1 - label) * torch.pow(euclidian_distance, 2) + label * torch.pow(torch.clamp(self.margin - euclidian_distance, min = 0.0), 2)
		contrastive_loss = torch.mean(contrastive_loss)
		return contrastive_loss