# File to add all the functions that run in individual processes
from pycromanager import Acquisition
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import torch.multiprocessing as tmp
from torch.utils.data import DataLoader
from utils import genericQueue, RNNQueue
from functools import partial
from queue import Empty
from utils import lockedNumpyArray
import torch.multiprocessing as tmp
import multiprocessing as mp
from narsil.liverun.utils import getPositionFileName, parsePositionsFile


def acquisition(segQueue, imgArrivalQueue, acqShutDownEvent, positionTimeTuple, timeWait):
	positionFilePath = getPositionFileName()
	positionData = parsePositionsFile(positionFilePath)
	pass

def plotter():
	pass

def segCellsAndChannels():
	pass

def deadAlive():
	pass

def putDummyImages():
	pass