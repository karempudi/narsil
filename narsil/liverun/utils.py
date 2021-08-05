# Utilites like locked arrays and datastructrues/classes that handle stuff outside
from torch.utils.data import DataLoader, Dataset, IterableDataset
import multiprocessing

"""
Gerenric queue for yielding objects from the queue in a safe way to be
processed by different processes piping into the queue and out of the
queue
"""
class genericQueue(IterableDataset):
	def __init__(self, queue):
		self.queue = queue
	
	def getNextImage(self):
		#print(f"Queue size in datalaoder: {self.queue.qsize()}")
		while self.queue.qsize() > 0:
			yield self.queue.get()
		return None

	def __iter__(self):
		return self.getNextImage()
	

class RNNQueue(IterableDataset):
	
	def __init__(self, queue, channelsWritePath):
		self.queue = queue
		self.lock = multiprocessing.Lock()
		self.channelsWritePath = channelsWritePath

	def getNextItem(self):
		# use locks and get the LSTM states that are written down in 
		# appropriate folders
		self.lock.acquire()
		while self.queue.qsize() > 0:
			channel = self.queue.get()
			# go get the lstm stacks, the release the locks


			self.lock.release()
			yield {'lstm': None, 'position': None, 'time': None}

		return None

	def __iter__(self):
		return self.getNextItem()
	
def lockedNumpyArray():
	pass
