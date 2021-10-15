import numpy as np
import glob
from skimage import io


class channelStackTrain(object):

    def __init__(self, phaseDirectoriesList, numUnrolls, fileformat='.tiff'):
        self.numUnrolls = numUnrolls
        self.phaseDirectoriesList = phaseDirectoriesList
        self.fileformat = fileformat
        self.dataSequences = []
        self.construct_dataset()

    def __getitem__(self, idx):
        # construct the images sequence, and state sequence for training
        imgSequenceFilenames, statesSequence = self.dataSequences[idx]
        imageStack = None

        for filename in imgSequenceFilenames:
            localimage = io.imread(filename).astype('float32')
            if imageStack is None:
                imageStack = localimage
            else:
                imageStack = np.dstack((imageStack, localimage))
        return {'imageSequence': np.expand_dims(imageStack, axis = 0),
                'statesSequence': statesSequence.astype('float32')}
    

    def __len__(self):
        return len(self.dataSequences)

    def construct_dataset(self):
        for directory in self.phaseDirectoriesList:
            filenames = [int(filename.split('.')[0].split('/')[-1]) for filename in glob.glob(directory + "*" + self.fileformat)]
            filenames.sort()
            sortedFilenames = [directory + str(filename) + self.fileformat for filenumber in filenames]
            statesFilename = glob.glob(directory + "*.npy")[0]
            states = np.load(statesFilename)
            # loop over and construct sequneces
            for i in range(0, len(sortedFilenames) - self.numUnrolls):
                imgSequenceFilenames = sortedFilenames[i: i+ self.numUnrolls + 1]
                sequencesStates = states[:, i:i+self.numUnrolls+1]

                self.dataSequences.append((imgSequenceFilenames, sequenceStates))


        