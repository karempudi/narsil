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
            localimage = (localimage - np.mean(localimage))/np.std(localimage)
            if imageStack is None:
                imageStack = localimage
            else:
                imageStack = np.dstack((imageStack, localimage))

        # only one image is present, so you will have to  add the time dimension that is
        # implicitly added in stacking images
        if len(imageStack.shape) == 2:
            imageStack = np.expand_dims(imageStack , -1)

        return {'imageSequence': np.expand_dims(imageStack, axis = 0),
                'statesSequence': statesSequence.astype('float32')}
    

    def __len__(self):
        return len(self.dataSequences)

    def construct_dataset(self):
        #print("Called dataset contruction")
        for directory in self.phaseDirectoriesList:
            filenames = [int(filename.split('.')[0].split('/')[-1]) for filename in glob.glob(directory + "*" + self.fileformat)]
            filenames.sort()
            sortedFilenames = [directory + str(filenumber) + self.fileformat for filenumber in filenames]
            statesFilename = glob.glob(directory + "*.npy")[0]
            states = np.load(statesFilename)
            #print(sortedFilenames)
            #print(states)
            # loop over and construct sequneces
            for i in range(0, len(sortedFilenames) - self.numUnrolls + 1):
                imgSequenceFilenames = sortedFilenames[i: i+ self.numUnrolls]
                sequenceStates = states[:, i:i+self.numUnrolls]

                self.dataSequences.append((imgSequenceFilenames, sequenceStates))


        