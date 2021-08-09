# Functions that help with growth rates
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
import seaborn as sb

np.seterr(divide='ignore', invalid='ignore')
def exp_growth_fit(x, a, b):
	return a * np.exp(-b * x)



class growthRatesPickles(object):
    """
    A class that will take in all the pickles from all the positions
    given and cleanup all rates, 
    Maybe we can add times
    """
    def __init__(self, analysisDir, speciesNames, speciesTitles, NoAbPositions = [], AbPositions = [], 
                trackingDirName = 'growthRates/', fileformat = '.pickle', nFrames=31,
                antibioticName='Nitro', antibioticConc = 4):

        self.analysisDir  = analysisDir
        self.speciesNames = speciesNames
        self.NoAbPositions = NoAbPositions
        self.AbPositions = AbPositions
        self.trackingDirName = trackingDirName
        self.fileformat = fileformat
        self.nFrames = nFrames
        self.antibioticName = antibioticName
        self.antibioticConc = antibioticConc
        self.speciesTitles = speciesTitles
        self.colorMap ={'Klebsiella': 'r', 'E.coli' : 'b', 'Pseudomonas': 'g', 'E.cocci' : 'm'}  

        self.NoAbDirectoriesList = []
        self.AbDirectoriesList = []
        for position in NoAbPositions:
            self.NoAbDirectoriesList.append(analysisDir + 'Pos' + str(position) + '/' + self.trackingDirName)

        for position in AbPositions:
            self.AbDirectoriesList.append(analysisDir + 'Pos' + str(position) + '/' + self.trackingDirName)

        self.NoAbPickleFilesList = []            
        self.AbPickleFilesList = []
        for directory in self.NoAbDirectoriesList:
            self.NoAbPickleFilesList.extend(glob.glob(directory + "*" + self.fileformat))

        for directory in self.AbDirectoriesList:
            self.AbPickleFilesList.extend(glob.glob(directory + "*" + self.fileformat))

        # puts together all the pickle files and then adds up the species dictionaries 
        self.constructGrowthRates()
        self.cleanGrowthRates()

    # Returns total number of channels analysed
    def __len__(self):
        return len(self.NoAbPickleFilesList) + len(self.AbPickleFilesList)

    def constructGrowthRates(self):
        self.NoAbSpeciesGrowthRates = {}
        self.AbSpeciesGrowthRates = {}
        self.NoAbPooledGrowthRates = []
        self.AbPooledGrowthRates = []

        self.NoAbCounts = {}
        self.AbCounts = {}

        for species in self.speciesNames:
            self.NoAbSpeciesGrowthRates[species] = []
            self.AbSpeciesGrowthRates[species] = []
            self.NoAbCounts[species] = 0
            self.AbCounts[species] = 0

        for filename in self.NoAbPickleFilesList:
            with open(filename, 'rb') as filehandle:
                data = pickle.load(filehandle)
            for key, value in data.items():
                if value != []:
                    self.NoAbCounts[key] += 1
                self.NoAbSpeciesGrowthRates[key] += value
                self.NoAbPooledGrowthRates += value
        
        self.NoAbPooledGrowthRates = np.array(self.NoAbPooledGrowthRates)
 
        for filename in self.AbPickleFilesList:
            with open(filename, 'rb') as filehandle:
                data = pickle.load(filehandle)
            for key, value in data.items():
                if value != []:
                    self.AbCounts[key] += 1
                self.AbSpeciesGrowthRates[key] += value
                self.AbPooledGrowthRates += value
        
        self.AbPooledGrowthRates = np.array(self.AbPooledGrowthRates)

    def __getitem__(self, idx):
        pass

    def cleanGrowthRates(self, frameRate=2):
        self.NoAbCleanGrowthRates = {}
        self.AbCleanGrowthRates = {}
        for species in self.speciesNames:
            self.NoAbCleanGrowthRates[species] = self.mean_std_counts(species, frameRate=frameRate, ab = False)
            self.AbCleanGrowthRates[species] = self.mean_std_counts(species, frameRate=frameRate, ab = True)

        noAb_pool_counts = np.zeros(shape=(self.nFrames,))
        Ab_pool_counts = np.zeros(shape=(self.nFrames,))
        noAb_pool_growth = np.zeros(shape=(self.nFrames,))
        Ab_pool_growth = np.zeros(shape=(self.nFrames,))
        noAb_pool_growth_dev = np.zeros(shape=(self.nFrames,))
        Ab_pool_growth_dev = np.zeros(shape=(self.nFrames,))
        for i in range(self.nFrames):
            noab_column = self.NoAbPooledGrowthRates[:, i]
            ab_column = self.AbPooledGrowthRates[:, i]
            if len(noab_column[noab_column != -1]) == 0:
                pass
            else:
                noAb_pool_growth[i] = np.mean(noab_column[np.logical_and(noab_column != -1, noab_column >=0)]/frameRate)
                noAb_pool_counts[i] = np.sum(np.logical_and(noab_column != -1, noab_column >=0))
                noAb_pool_growth_dev[i] = np.std(noab_column[np.logical_and(noab_column != -1, noab_column >=0)]/frameRate)
            
            if len(ab_column[ab_column != -1]) == 0:
                pass
            else:
                Ab_pool_growth[i] = np.mean(ab_column[np.logical_and(ab_column != -1, ab_column >= 0)]/frameRate)
                Ab_pool_counts[i] = np.sum(np.logical_and(ab_column != -1, ab_column >= 0))
                Ab_pool_growth_dev[i] = np.std(ab_column[np.logical_and(ab_column != -1, ab_column >= 0)]/frameRate)
        
        self.NoAbCleanPooledGrowthRates = (noAb_pool_growth, noAb_pool_growth_dev, noAb_pool_counts)
        self.AbCleanPooledGrowthRates = (Ab_pool_growth, Ab_pool_growth_dev, Ab_pool_counts)

            

    # return tuple (mean, std deviation, counts)
    def mean_std_counts(self, species, frameRate = 2, ab = False):
        if ab == False:
            speciesGrowthRates = np.array(self.NoAbSpeciesGrowthRates[species])
            counts_t_species = np.zeros(shape=(self.nFrames,))
            growth_t_species = np.zeros(shape=(self.nFrames,))
            growth_dev_t_species = np.zeros(shape=(self.nFrames,))
            for i in range(self.nFrames):
                column = speciesGrowthRates[:, i]
                if len(column[column != -1]) == 0:
                    pass
                else:
                    growth_t_species[i] = np.mean(column[np.logical_and(column != -1, column >= 0)]/frameRate)
                    counts_t_species[i] = np.sum(np.logical_and(column != -1, column >=0))
                    growth_dev_t_species[i]  = np.std(column[np.logical_and(column != -1, column >= 0)]/frameRate)
            
            return growth_t_species, growth_dev_t_species, counts_t_species
        elif ab == True:
            speciesGrowthRates = np.array(self.AbSpeciesGrowthRates[species])
            counts_t_species = np.zeros(shape=(self.nFrames,))
            growth_t_species = np.zeros(shape=(self.nFrames,))
            growth_dev_t_species = np.zeros(shape=(self.nFrames,))
            for i in range(self.nFrames):
                column = speciesGrowthRates[:, i]
                if len(column[column != -1]) == 0:
                    pass
                else:
                    growth_t_species[i] = np.mean(column[np.logical_and(column != -1, column >=0)]/frameRate)
                    counts_t_species[i] = np.sum(np.logical_and(column != -1, column >=0))
                    growth_dev_t_species[i]  = np.std(column[np.logical_and(column != -1, column >=0)]/frameRate)
            
            return growth_t_species, growth_dev_t_species, counts_t_species
 
    def getGrowthRates(self, species):
        if species not in self.speciesNames:
            return None
        species_noab = self.NoAbCleanGrowthRates[species]
        species_ab = self.AbCleanGrowthRates[species]
        normalized_growth_rates = species_ab[0]/species_noab[0]
        species_err_no_ab = species_noab[1]/species_noab[0]/np.sqrt(species_noab[2])
        species_err_ab = species_ab[1]/species_noab[0]/np.sqrt(species_ab[2])

        return (normalized_growth_rates, species_err_no_ab, species_err_ab)


    def getSpeciesStatisitcs(self):
        pass

    def plotGrowthRates(self, species):
        pass

    # color scheme is a dict of rgb values for each species
    def plotSpeciesWiseAndPooled(self, colorscheme, speciesFullName = { 'Klebsiella': "K. pneumoniae", 
        "E.coli": "E.coli", "Pseudomonas": "P.aeruginosa", "E.cocci": "E.faecalis"},
        ignore = []):
        sb.set_style("white")
        fig, ax = plt.subplots(nrows=1, ncols=1)
        #plt.rcParams['axes.linewidth'] = 1.2
        for i in range(len(self.speciesNames)):
            
            if self.speciesNames[i] in ignore:
                continue
            species_noab = self.NoAbCleanGrowthRates[self.speciesNames[i]]
            species_ab = self.AbCleanGrowthRates[self.speciesNames[i]]
            normalized_growth_rates = species_ab[0]/species_noab[0]
            species_err_no_ab = species_noab[1]/species_noab[0]/np.sqrt(species_noab[2])
            species_err_ab = species_ab[1]/species_noab[0]/np.sqrt(species_ab[2])

            ax.plot(range(0, 2*self.nFrames, 2), normalized_growth_rates, color=colorscheme[self.speciesNames[i]], label=speciesFullName[self.speciesNames[i]]+ ' Treatment')
            ax.fill_between(range(0, 2 * self.nFrames, 2), normalized_growth_rates - species_err_ab, normalized_growth_rates + species_err_ab,
                                  alpha = 0.4, color= colorscheme[self.speciesNames[i]], linestyle='--', linewidth=2)
         
         
        # plot some pooled curve here to see if you can differentiate
    
        noAb_pool = self.NoAbCleanPooledGrowthRates
        Ab_pool = self.AbCleanPooledGrowthRates
        normalized_pool = Ab_pool[0]/noAb_pool[0]
        noAb_pool_err = noAb_pool[1]/noAb_pool[0]/np.sqrt(noAb_pool[2])
        Ab_pool_err = Ab_pool[1]/noAb_pool[0]/np.sqrt(Ab_pool[2])

        # Normalized growth rates no antibiotic
        ax.plot(range(0, 2*self.nFrames, 2), [1] * self.nFrames, 'k', label='No Species Id Reference')

        # standard error of the normalized values
        ax.fill_between(range(0, 2 * self.nFrames, 2), [1]* self.nFrames - noAb_pool_err, 
                    [1] * self.nFrames + noAb_pool_err, alpha=0.4, color='k', linestyle='--', linewidth=2)

        # standard deviation of the normalized values
        #ax[1, 1].fill_between(range(0, 2 * self.nFrames, 2), [1]* self.nFrames - noAb_pool[1]/noAb_pool[0], 
        #            [1] * self.nFrames + noAb_pool[1]/noAb_pool[0], alpha=0.2, color='b')

        # normalized growth rates - with antibiotic
        ax.plot(range(0, 2*self.nFrames, 2), normalized_pool, 'k:', label='No species Id Treatment')
        ax.fill_between(range(0, 2 * self.nFrames, 2), normalized_pool - Ab_pool_err, 
                    normalized_pool + Ab_pool_err, alpha=0.4, color='k', linestyle=':', linewidth=2)
        #ax[1, 1].fill_between(range(0, 2 * self.nFrames, 2), normalized_pool - Ab_pool[1]/noAb_pool[0], 
        #            normalized_pool + Ab_pool[1]/noAb_pool[0], alpha=0.2, color='r')
        #ax.set_ylim([0, 1.4])
        #ax.set_ylabel("Growth Rate (normalized)")
        #ax.set_xlabel("Time(min)")
        #ax.legend(loc='lower left')


        
        ax.set_xlim([8, 2 * self.nFrames - 2])
        #ax.set_ylabel("Growth Rate (normalized)")
        #ax.set_title(f"{self.antibioticName} {self.antibioticConc}" + r'$\mu g/ml$ ' + " All species")
        plt.xticks(fontsize=12, weight='bold')
        ax.set_ylim([0, 1.4])
        plt.yticks(fontsize=12, weight='bold')
        ax.legend(loc='best',fontsize='large',framealpha=0.3)
        #ax.set_xlabel("Time(min)")
 


    # ignore is a list of species you can ignore while plotting
    def plotAllFigures(self, std_err=True, std_dev=True, ignore = [], ylim=1.6, speciesFullName = { 'Klebsiella': "K. pneumoniae", 
        "E.coli": "E.coli", "Pseudomonas": "P.aeruginosa", "E.cocci": "E.faecalis"}):
        nrows = 2
        ncols = 4
        sb.set_style("white")
        fig, ax = plt.subplots(nrows=2, ncols=4)
        for i in range(len(self.speciesNames)):
            # species_noab[0] -- mean growth rates
            # species_noab[1] -- std dev growth rates
            # species_noab[2] -- no of tracks at any timepoint
            if self.speciesNames[i] in ignore:
                continue
            species_noab = self.NoAbCleanGrowthRates[self.speciesNames[i]]
            species_ab = self.AbCleanGrowthRates[self.speciesNames[i]]
            normalized_growth_rates = species_ab[0]/species_noab[0]
            species_err_no_ab = species_noab[1]/species_noab[0]/np.sqrt(species_noab[2])
            species_err_ab = species_ab[1]/species_noab[0]/np.sqrt(species_ab[2])

            # plotting std_err and std_dev of normalized growth rates
            
            # Normalized growth rates no antibiotic
            ax[0, i].plot(range(0, 2*self.nFrames, 2), [1] * self.nFrames, 'b-', label='Reference')

            # standard error of the normalized values

            ax[0, i].fill_between(range(0, 2 * self.nFrames, 2), [1]* self.nFrames - species_err_no_ab, 
                        [1] * self.nFrames + species_err_no_ab, alpha=0.4, color='b', linestyle='--', linewidth=2)

            # standard deviation of the normalized values
            #ax[0, i].fill_between(range(0, 2 * self.nFrames, 2), [1]* self.nFrames - species_noab[1]/species_noab[0], 
            #            [1] * self.nFrames + species_noab[1]/species_noab[0], alpha=0.2, color='b')

            # normalized growth rates - with antibiotic
            ax[0, i].plot(range(0, 2*self.nFrames, 2), normalized_growth_rates, 'r-', label='Treatment')
            ax[0, i].fill_between(range(0, 2 * self.nFrames, 2), normalized_growth_rates - species_err_ab, 
                        normalized_growth_rates + species_err_ab, alpha=0.4, color='r', linestyle='--', linewidth=2)
            #ax[0, i].fill_between(range(0, 2 * self.nFrames, 2), normalized_growth_rates - species_ab[1]/species_noab[0], 
            #            normalized_growth_rates + species_ab[1]/species_noab[0], alpha=0.2, color='r')
            ax[0, i].set_xlim([8, 2*self.nFrames - 2])
            ax[0, i].set_ylim([0, 1.4])
            ax[0, i].set_title(f"{self.antibioticName} {self.antibioticConc}" + r'$\mu g/ml$ ' + f"{speciesFullName[self.speciesNames[i]]}")
            ax[0, i].set_ylabel("Growth Rate (normalized)")
            ax[0, i].set_xlabel("Time(min)")
            ax[0, i].legend(loc='lower left')


        # plot relative curves based on species

        ax[1, 0].plot(range(0, 2 * self.nFrames, 2), [1] * self.nFrames, color='k', label='Reference')
        for i in range(len(self.speciesNames)):
            if self.speciesNames[i] in ignore:
                continue
            species_noab = self.NoAbCleanGrowthRates[self.speciesNames[i]]
            species_ab = self.AbCleanGrowthRates[self.speciesNames[i]]
            normalized_growth_rates = species_ab[0]/species_noab[0]
            species_err_ab = species_ab[1]/species_noab[0]/np.sqrt(species_ab[2])

            ax[1, 0].plot(range(0, 2*self.nFrames, 2), normalized_growth_rates, color=self.colorMap[self.speciesNames[i]], label=speciesFullName[self.speciesNames[i]])
            ax[1, 0].fill_between(range(0, 2 * self.nFrames, 2), normalized_growth_rates - species_err_ab, normalized_growth_rates + species_err_ab,
                                  alpha = 0.4, color= self.colorMap[self.speciesNames[i]], linestyle='--', linewidth=2)
            
        ax[1, 0].set_xlim([8, 2 * self.nFrames - 2])
        ax[1, 0].set_ylabel("Growth Rate (normalized)")
        ax[1, 0].set_title(f"{self.antibioticName} {self.antibioticConc}" + r'$\mu g/ml$ ' + " All species")
        ax[1, 0].set_ylim([0, 1.4])
        ax[1, 0].legend(loc='best',fontsize='medium',framealpha=0.3)
        ax[1, 0].set_xlabel("Time(min)")
        




        # plot number of cells available at each time point
        for i in range(len(self.speciesNames)):
            if self.speciesNames[i] in ignore:
                continue
            
            species_noab = self.NoAbCleanGrowthRates[self.speciesNames[i]]
            species_ab = self.AbCleanGrowthRates[self.speciesNames[i]]

            ax[1, 2].plot(range(0, 2 *self.nFrames, 2), species_noab[2], color=self.colorMap[self.speciesNames[i]], label=speciesFullName[self.speciesNames[i]]+ " Reference")
            ax[1, 2].plot(range(0, 2 * self.nFrames, 2), species_ab[2], color=self.colorMap[self.speciesNames[i]], linestyle='--', label=speciesFullName[self.speciesNames[i]]+ " Treatment")
        
        ax[1, 2].legend(loc='upper left', fontsize='x-small')
        ax[1, 2].set_xlim([8, 2*self.nFrames -2])
        ax[1, 2].set_xlabel("Time(min)")
        ax[1, 2].set_ylabel("Number of cells")


        # plot some pooled curve here to see if you can differentiate
    
        noAb_pool = self.NoAbCleanPooledGrowthRates
        Ab_pool = self.AbCleanPooledGrowthRates
        normalized_pool = Ab_pool[0]/noAb_pool[0]
        noAb_pool_err = noAb_pool[1]/noAb_pool[0]/np.sqrt(noAb_pool[2])
        Ab_pool_err = Ab_pool[1]/noAb_pool[0]/np.sqrt(Ab_pool[2])

        # Normalized growth rates no antibiotic
        ax[1, 1].plot(range(0, 2*self.nFrames, 2), [1] * self.nFrames, 'b-', label='Reference')

        # standard error of the normalized values

        ax[1, 1].fill_between(range(0, 2 * self.nFrames, 2), [1]* self.nFrames - noAb_pool_err, 
                    [1] * self.nFrames + noAb_pool_err, alpha=0.4, color='b', linestyle='--', linewidth=2)

        # standard deviation of the normalized values
        #ax[1, 1].fill_between(range(0, 2 * self.nFrames, 2), [1]* self.nFrames - noAb_pool[1]/noAb_pool[0], 
        #            [1] * self.nFrames + noAb_pool[1]/noAb_pool[0], alpha=0.2, color='b')

        # normalized growth rates - with antibiotic
        ax[1, 1].plot(range(0, 2*self.nFrames, 2), normalized_pool, 'r-', label='Treatment')
        ax[1, 1].fill_between(range(0, 2 * self.nFrames, 2), normalized_pool - Ab_pool_err, 
                    normalized_pool + Ab_pool_err, alpha=0.4, color='r', linestyle='--', linewidth=2)
        #ax[1, 1].fill_between(range(0, 2 * self.nFrames, 2), normalized_pool - Ab_pool[1]/noAb_pool[0], 
        #            normalized_pool + Ab_pool[1]/noAb_pool[0], alpha=0.2, color='r')
        ax[1, 1].set_xlim([8, 2*self.nFrames - 2])
        #ax[0, i].set_ylim([0, 5])
        ax[1, 1].set_title(f"{self.antibioticName} {self.antibioticConc}" + r'$\mu g/ml$ ' + "Pooled. No species ID")
        ax[1, 1].set_ylim([0, 1.4])
        ax[1, 1].set_ylabel("Growth Rate (normalized)")
        ax[1, 1].set_xlabel("Time(min)")
        ax[1, 1].legend(loc='lower left')



        # plot channel counts to see if the species were loaded equally on both sides
        labels = [speciesFullName[species] for species in self.speciesNames]
        noAbCounts = []
        AbCounts = []
        for species in self.speciesNames:
            noAbCounts.append(self.NoAbCounts[species])
            AbCounts.append(self.AbCounts[species])
        
        x = np.arange(len(labels))
        width = 0.35
        ax[1, 3].bar(x - width/2, noAbCounts, width, label='Reference', color='b')
        ax[1, 3].bar(x + width/2, AbCounts, width, label='Treatment', color='r')
        ax[1, 3].set_ylabel("Number of channels")
        ax[1, 3].set_title("Species vs Number of channels")
        ax[1, 3].set_xticks(x)
        ax[1, 3].set_xticklabels(labels)
        ax[1, 3].legend()
        print(labels)



        plt.subplots_adjust(hspace=0.207, wspace=0.245, top=0.960, bottom=0.060, left=0.042, right=0.986)
