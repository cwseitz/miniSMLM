from skimage.io import imsave
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

config_path = 'run_track_2d.json'
with open(config_path, 'r') as f:
    config = json.load(f)

prefixes = ['230909_Hela_Fixed-Sub']

for prefix in prefixes:
    print("Processing " + prefix)
    path = config['analpath'] + prefix + '/' + prefix + '_spots.csv'
    savepath = '-tracked.'.join(path.split('.'))
    tracker = Tracker2D(config)
    spots = pd.read_csv(path)
    linked = tracker.link(spots)
    imsd = tracker.imsd(linked)
    imsd.to_csv(savepath)
