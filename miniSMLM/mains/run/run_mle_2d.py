from pipes import PipelineMLE2D
from miniSMLM.utils import SMLMDataset
from miniSMLM.utils import KDE, make_animation
from skimage.io import imsave
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

config_path = 'run_all_2d.json' #replace with path to your config
with open(config_path, 'r') as f:
    config = json.load(f)

prefixes = ['230909_Hela_Fixed-Sub']

for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    pipe = PipelineMLE2D(config,dataset)
    pipe.localize(plot_spots=False,plot_fit=False,tmax=1000)
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    make_animation(dataset.stack,spots)
    render = KDE(spots).forward(sigma=2.0)
    imsave(config['analpath']+prefix+'/'+prefix+'-kde',render)
