# The run file for running the realtime processing
from matplotlib.pyplot import plot
import time
import torch.multiprocessing as tmp
import multiprocessing as python_mulitprocessing
from narsil.liverun.processes import acquisition, segCellsAndChannels, deadAlive, plotter
import psycopg2 as pg
import sys

"""
Initialize the database for the experiment
"""
def initializeDatabase(databaseParameters):
	dbname = databaseParameters['dbname']
	dbuser = databaseParameters['dbuser']
	dbpassword = databaseParameters['dbpassword']

	# usually is ['arrival', 'segmented', 'deadAlive']
	tables = databaseParameters['tables']

	con = None

	try:
		con = pg.connect(database=dbname, user=dbuser, password=dbpassword)
		cur = con.cursor()

		# commit to database immediately
		con.autocommit = True
		# loop over and check if tables exist and clean them up
		cur.exeute("""SELECT table_name FROM information_schema.tables
					WHERE table_schema = 'public'""")

		rows = cur.fetchall()
		for row in rows:
    		if row in tables:
				cur.execute("DROP TABLE IF EXISTS (%s)", (row,))
		
		print(f"Clean up all of {tables} to start new experiment ....")

		for table in tables:
			if table == 'arrival':
				cur.execute("""CREATE TABLE arrival
						(id SERIAL PRIMARY KEY, time TIMESTAMP, position INT)
						""")
			elif table == 'segmented':
    			cur.execute("""CREATE TABLE segmented
						(id SERIAL PRIMARY KEY, time TIMESTAMP, position INT, 
						segmentedImagePath VARCHAR, rawImagePath VARCHAR, locations BYTEA)
						""")
			elif table == 'deadAlive':
				cur.execute("""CREATE TABLE deadAlive
						(id SERIAL PRIMARY KEY, time TIMESTAMP, position INT,
						channelNumber INT, status BYTEA)
						""")

	except pg.DatabaseError as e:
		print(f"Error: {e}")
		sys.exit(1)
	finally:
		if con:

"""
Set up the queues and start the experiment ...
"""
def startHeteroExperiment():
	print("Welcome to real-time processing pipeline ... :) :)")
	print("Setting up the queuing systems ... ")
	start = time.time()
	jobs = []


	# Parameters:
	acquisitionParameters = {
		"imageSizeParameters": 0

	}	

	segmentationParameters = {

	}

	databaseLoggingParameters = {
		'dbname': 'Expt number',
		'dbuser': "postgres",
		'dbpassword': "password",
		'tables': ['arrival', 'segmented', 'deadAlive']
	}

	deadAliveParameters = {

	}

	plottingParameters = {

	}

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
							args=(segQueue, imgArrivalQueue, acqShutDownEvent, acquisitionParameters),
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
