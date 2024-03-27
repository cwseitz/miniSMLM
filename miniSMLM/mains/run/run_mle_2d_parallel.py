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

config_path = 'miniSMLM-main/miniSMLM/mains/run/run_mle_2d_parallel.json' #replace with path to your config
tmax = 10 # debug variable, will be removed

with open(config_path, 'r') as f:
    config = json.load(f)

prefixes = ['LIV-U2OS-STORM-240102_live cell-LIV__1_MMStack_Default.ome']

# helper function for using multiprocess module to dill instead of pickle
def job(frame):
    values = frame.localize()
    return values

for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix) 

    frames = [Localizer(i, config, dataset) for i in range(tmax)] # need to call p.map from main, name == foo does not work in ParallelLoc
    outputs = []
    if __name__ == '__main__': # maybe not necessary in Linux, will check
        with Pool() as p: 
            outputs = p.map(job, frames) 
    
    print(outputs) # to be done: remove extra column labels from dataframes before appending, save as csv

    # spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    # make_animation(dataset.stack,spots)
    # render = KDE(spots).forward(sigma=2.0)
    # imsave(config['analpath']+prefix+'/'+prefix+'-kde.tif',render)
