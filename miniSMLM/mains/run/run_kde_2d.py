import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from miniSMLM.utils import KDE
from tifffile import imread
from skimage.io import imsave

with open('run_kde_2d.json', 'r') as f:
    config = json.load(f)

prefixes = []

plot=False
for n,prefix in enumerate(prefixes):
    print("Processing " + prefix)
    spots = pd.read_csv(config['datapath'] + prefix + '_spots.csv')
    render = KDE(spots).get_kde(sigma=2.0)

    if plot:
        fig,ax=plt.subplots()
        ax.imshow(render,cmap='gray',vmin=0,vmax=0.2)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

        fig,ax=plt.subplots()
        ax.scatter(spots['y_mle'],spots['x_mle'],s=1,color='black',marker='x')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

        plt.show()

    imsave(config['analpath']+prefix+'-kde.tif',render)
