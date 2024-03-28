import sys
sys.path.append('miniSMLM-main/') 
from pipes import Localizer
from miniSMLM.utils import SMLMDataset
from miniSMLM.utils import KDE, make_animation
from skimage.io import imsave
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from multiprocessing import Pool

config_path = 'miniSMLM-main/miniSMLM/mains/run/run_mle_2d_parallel.json' # replace with path to your config
prefixes = ['LIV-U2OS-STORM-240102_live cell-LIV__1_MMStack_Default.ome'] # replace with tif file names without the .tif extension

with open(config_path, 'r') as f:
    config = json.load(f)

for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix) 

    # localization call to helper function and saving of mapped MLEs
    frames = range(min(config['tmax'], len(dataset.stack))) # would call len(dataset) but line 24 in utils.dataset.py assigns 4 values to the 3 sized stack, not sure why
    outputs = []
    if __name__ == '__main__': # failsafe CPU protection
        with Pool(config['processes']) as p: 
            outputs = p.map(job, frames) 
    Localizer.save(outputs, prefix, config['analpath'])

    # creation and saving of super res image
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    make_animation(dataset.stack,spots)
    render = KDE(spots).forward(sigma=2.0)
    imsave(config['analpath']+prefix+'/'+prefix+'-kde.tif',render)

# helper function for multiprocessing
def job(n):
    frame = Localizer(n, config, dataset)
    spots = frame.localize()
    return spots
