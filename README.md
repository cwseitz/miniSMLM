# miniSMLM

Minimal repository for analyzing localization microscopy images for super-resolution. Useful for teaching purposes and basic SR analysis

## Basic installation

This code has been tested on Ubuntu 22.04.3 LTS. [Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/) is required to create a virtual environment. All of the necessary dependencies are specified in the conda environment file ```miniSMLM.yml```. Assuming you have Anaconda already installed on your machine, run
 
``` 
conda env create -f /path/to/miniSMLM.yml
conda activate miniSMLM
```  
Then change to the root directory of miniSMLM, which contains ```setup.py```. Run 

``` 
pip install -e .
```  

This will install miniSMLM in development mode

## The coordinate system

All coordinates throughout the code (blob detection and fitting) are in *image* coordinates where x represents the row and y represents the column, measured from the top left of the image. The blob_log() function in scikit-image uses image coordinates by default. Therefore, in isologlike2d() we use 'ij' indexing in numpy's meshgrid to preserve image coordinates.

When using matplotlib's scatter() to show detected points, plotting code must transpose coordinates and invert the y axis. 


