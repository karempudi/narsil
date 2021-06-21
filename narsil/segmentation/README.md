## How to train a network

#### Data Directories structure

One of the goals of this project is to be able to build the ultimate segmentation
model one would ever need for segmenting phase contrast images.
In the project related to the first paper that uses this package, we develop reasonably 
good segmentation models for 4 species E.*coli*, P.*aeruginosa*, E.*faecalis*, K.*pnemonia* 

```
+-- trainingDataDir
|	+-- ecoli
|		+-- phase_train
|		+-- mask_train
|		+-- weights_train
|		+-- phase_validation
|		+-- mask_validation
|		+-- weights_validation
|	+-- pseudomonas
|		+-- phase_train
|		+-- mask_train
|		+-- weights_train
|		+-- phase_validation
|		+-- mask_validation
|		+-- weights_validation
+--
```
Each of the train and mask directories contain phase contrast images and corresponding 
binary segmentation masks. Weights directories contain weight maps that are used for
seperating cells, as described in the original [U-net](https://arxiv.org/pdf/1505.04597.pdf) paper. Functions for generating weight maps from binary segmentation mask is ..


#### Sample code for training your network
```python3
from narsil.segmentation.modelDev import trainNet
from torchvision import transforms
from narsil.utils.transforms import changedtoPIL, randomCrop, randomRotation, randomAffine, normalize, toTensor, addnoise

dataDir = '/mnt/sda1/Praneeth/trainingData/'

species = ['ecoli', 'pseudomonas']

modelSavePath = '../segModels/test_pseudo.pth'


imgtransforms = transforms.Compose([changedtoPIL(), randomCrop(320), 
                                    randomRotation([-20,20]),
                                randomAffine((0.75, 1.25), [-30, 30, -30, 30]),
                                toTensor(),
                                normalize(),
                                addnoise(std=0.15)])

modelParameters = {
    'netType': 'big',
    'transposeConv': True,
    'device': "cuda:1",
    'includeWeights': True
}

optimizationParameters = {
    'learningRate': 1e-4,
    'nEpochs': 1,
    'batchSize': 8,
    'cores': 4,
    'schedulerStep': 2,
    'schedulerGamma': 0.5
}

net = trainNet(dataDir, species, imgtransforms, modelParameters, optimizationParameters)

net.train()

net.save(modelSavePath)

```

Saved model has a represenation of what transforms were used to train, what species were used,
and the optimization and model parameters, so you can look back and compare models for the ones
that provide better performance.



#### Different Modules in the segmentation sub-package


#### Running segmentation on one directory containing .tiff files

Directory structure:


```
segmentationParameters = {
    'device': "cuda:1",
    'fileformat' : ".tiff",
    'batch_size': 1,
    'segmentationThreshold': 0.9,
    'flipPositions': range(201, 230),
    'phasePreset': 'phaseFast',
    'addNoise': False,
    'getChannelLocations': True,
    'channelSegThreshold': 0.8,
    'numChannels': 16,
    'dilateAfterSeg': True,
    'saveSeg': True,
    'savePhase': False,
    'backgroundArtifactClean': False,
    'minBackgroundOnPhase': 32000,
    'channelCuttingParameters' : {
        'channel_min': 300,
        'channel_max': 700,
        'channel_sum': 300,
        'histPeaksDistance': 35,
        'minBarcodeDistance': 80,
        'firstBarcodeIndex', 10,
        'numChannels': 16,
    }
}
```


