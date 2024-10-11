from pipes import PipelineMLE2D
from miniSMLM.utils import SMLMDataset
from miniSMLM.utils import KDE, make_animation
from skimage.io import imsave
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

config_path = 'C:\\Users\\ishaa\\Documents\\SuperResolution\\miniSMLM\\miniSMLM\\mains\\run\\run_mle_2d.json'
with open(config_path, 'r') as f:
    config = json.load(f)

prefixes = [
    "C:\\Users\\ishaa\\Documents\\SuperResolution\\zhaoyuan\\U2OS-Green\\U2OS-Green_8\\U2OS-Green_8_MMStack_Default.ome"
]

if __name__ == '__main__':
    for prefix in prefixes:
        print("Processing " + prefix)
        dataset = SMLMDataset(config['datapath']+prefix,prefix)
        pipe = PipelineMLE2D(config,dataset)
        pipe.localize(plot_spots=False,plot_fit=False,tmax=1000)
        spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
        make_animation(dataset.stack,spots)
        render = KDE(spots).forward(sigma=2.0)
        imsave(config['analpath']+prefix+'/'+prefix+'-kde.tif',render)
