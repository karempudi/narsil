# contains Utils like GUI needed to create training data for the deadAliveNet

import glob
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from skimage import io


class createTraningData(object):

    def __init__(self, segmentedDir=None, phaseDir=None, fileformat='.tiff',
                 saveName='states.npy'):

        self.segmentedDir = segmentedDir
        self.phaseDir = phaseDir
        self.savedFileName = segmentedDir + saveName
        self.fileformat = fileformat
        if self.segmentedDir != None:
            self.indices = [int(filename.split('.')[0].split('/')[-1]) for filename in 
                            glob.glob(self.segmentedDir + "*" + self.fileformat)]
            self.indices.sort()
        else:
            self.indices = range(0, 30)
        
        # There are 6 states, whose probabilities you predict using the newtork
        # The states are MOVING, NOT-MOVING, PARTLY-DEAD, ALLDEAD, NOCELLS, CELLSVANISHED
        self.savedStates = np.full((6, len(self.indices)), False, dtype=bool)
        self.savedStates[0,:] = True

        ###################################################
        ################# GUI stuff #######################
        ###################################################
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, num=str(self.segmentedDir))
        plt.subplots_adjust(left=0.25, bottom=0.25)
        self.axcolor= 'lightgoldenrodyellow'
        # set some images
        self.pltSegImg = self.ax1.imshow(self.__getitem__(0)['segImage'], cmap='gray')
        self.pltPhaseImg = self.ax2.imshow(self.__getitem__(0)['phaseImage'], cmap='gray')

        # slider object to scroll
        self.axtime = plt.axes([0.25, 0.15, 0.55, 0.03], facecolor=self.axcolor)
        plt.subplots_adjust(bottom=0.25)
        length = len(self.indices)
        self.currentTimePoint = 0
        self.stime = Slider(self.axtime, 'Time', 0, length, valstep=1)
        self.stime.on_changed(self.updateFig)

        # Buttons to save and reset
        self.resetax = plt.axes([0.65, 0.025, 0.1, 0.03])
        self.saveax = plt.axes([0.8, 0.025, 0.1, 0.03])
        self.resetButton = Button(self.resetax, 'Reset', color=self.axcolor, hovercolor='0.975')
        self.saveButton = Button(self.saveax, 'Save', color=self.axcolor, hovercolor='0.975')
        self.resetButton.on_clicked(self.reset)
        self.saveButton.on_clicked(self.save)

        # Radio buttons to set states like blobs disapper, blobs move

        self.states = ['Moving', 'NotMoving', 'PartlyDead', 'AllDead', 'NoCells', 'CellsVanished']
        self.checkax = plt.axes([0.025, 0.6, 0.15, 0.15], faceolor=self.axcolor)
        self.checkButtons = CheckButtons(self.checkax, self.states, [True, False, False, False, False, False])
        self.checkButtons.on_clicked(self.applyStates)

        plt.pause(0.1)

    def __getitem__(self, idx):

        segFileName = self.segmentedDir + str(self.indices[idx]) + self.fileformat
        phaseFileName = self.phaseDir + str(self.indices[idx]) + self.fileformat

        return {
            'segImage': io.imread(segFileName),
            'phaseImage': io.imread(phaseFileName)
        }
    
    def __len__(self):
        return len(self.indices)

    def updateFig(self, val):
        index = int(self.stime.val)
        if index in self.indices:
            self.pltSegImg.set_data(self.__getitem__(index))['segImage']
            self.pltPhaseImg.set_data(self.__getitem__(index))['phaseImage']
            self.currentTimePoint = index

    def applyStates(self, label):
        row_index = self.states.index(label)
        self.savedStates[row_index, self.currentTimePoint:] = self.checkButtons.get_status()[row_index]

    def save(self, save):
        print(self.savedStates)
        print("-----")
        with open(self.savedFileName, 'wb') as f:
            np.save(f, self.savedStates)
            print(f"{self.savedStates} saved :)")

    def reset(self, reset):
        self.savedStates = np.full((6, len(self.indices)), False, dtype=bool)
        self.savedStates[0, :] = True
