# Narsil

A package built for analyzing time-lapse microscopy data of prokaryotes.
Most time-lapse data analysis involves extracting information about the cells in the data.

The primary goal of this package is to build primitives that helps in building your own pipelines for image analysis. The basic primitives involve segmentation, tracking and operations on tracks.

As a start, the pipeline is built to work with mother-machine datasets containing multiple-species of prokaryotes growing.

The secondary goal is to evolve this pipeline to be able to work in cooperation with microscopes to run the analysis while the experimental data is being acquired. We aim to close the gap between experiments and analysis to be able to run closed-loop experiments in the future.

The algorithms we use for segmentation and tracking are mainly based on deep learning. The package has functions to help train your own networks as well as run ours.

The advent of mother-machine like devices in microbiology promotes the need for standardization of analysis tools i.e., make things like segmenting a tiff stack, or tracking cells in a tiff stack as easy as possible.


### To install the package

First clone the repository and run the following commands.

```
pip install .
```
or

```
python3 setup.py install
```

### Package structure


