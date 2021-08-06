# The run file for running the realtime processing
from matplotlib.pyplot import plot
import time
import torch.multiprocessing as tmp
import multiprocessing as python_mulitprocessing
from narsil.liverun.processes import acquisition, segCellsAndChannels, deadAlive, plotter

"""
Set up the queues and start the experiment ...
"""
def startHeteroExperiment():
	print("Welcome to real-time processing pipeline ... :) :)")
	print("Setting up the queuing systems ... ")
	start = time.time()
	jobs = []

	# Shutdown event for the acquisition process
	acqShutDownEvent = tmp.Event()

	# segmentation Queue
	segQueue = tmp.Queue()
	segShutDownEvent = tmp.Event()

	# cutting and processing channels in the images
	channelProcessQueue = tmp.Queue()
	channelProcessShutDownEvent = tmp.Event()

	# status Queue for plotting, 
	imgArrivalQueue = python_mulitprocessing.Queue()
	imgProcessQueue = tmp.Queue()
	channelTrackQueue = tmp.Queue()
	deadAliveQueue = tmp.Queue()
	exptShutDownEvent = tmp.Event()

	nprocs = tmp.cpu_count()
	print(f"Number of CPU cores: {nprocs}")

	acquisitionProcess = tmp.Process(target=acquisition, 
							args=(segQueue, imgArrivalQueue, acqShutDownEvent, positionTimeTuple, timeWait),
							name="Acquisition")

	segAndChannelDetProcess = tmp.Process(target=segCellsAndChannels,
							args=(segQueue, segShutDownEvent, imgProcessQueue, channelProcessQueue,
								  segBatchSize, exptSavePath),
							name="SegCellsAndChannels")

	deadAliveProcess = tmp.process(target=deadAlive,
							args=(channelProcessQueue, channelProcessShutDownEvent, exptSavePath, saveOptions,
								  channeltrackQueue, deadAliveQueue, channelProcessBatchSize),
							name="DeadAlive")

	plotterProcess = tmp.process(target=plotter, 
							args=(imgArrivalQueue, imgProcessQueue, channelTrackQueue, deadAliveQueue,
								  exptShutDownEvent, positionTimeTuple, numChannels),
							name="Plotter")

	jobs.append(acquisitionProcess)
	jobs.append(segAndChannelDetProcess)
	jobs.append(deadAliveProcess)
	jobs.append(plotterProcess)

	try:
		# start all processes
		acquisitionProcess.start()
		segAndChannelDetProcess.start()
		deadAliveProcess.start()
		plotterProcess.start()

		# wait for them to finish
		acquisitionProcess.join()
		segAndChannelDetProcess.join()
		deadAliveProcess.join()
		plotterProcess.join()

		segQueue.cancel_join_thread()
		channelProcessQueue.cancel_join_thread()

	except KeyboardInterrupt:
		acqShutDownEvent.set()
		segShutDownEvent.set()
		channelProcessShutDownEvent.set()

		acquisitionProcess.join(5)
		segAndChannelDetProcess.join(5)
		deadAliveProcess.join(5)
		plotterProcess.join(5)

		print(f"Segmentation Queue is going to shut down has {segQueue.qsize()} images")

	if acquisitionProcess.is_alive():
		acquisitionProcess.kill()

	end = time.time()
	print(f"Duration of the analysis:: {end - start}s")


if __name__ == "__main__":
	startHeteroExperiment()
