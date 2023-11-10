from skimage.io import imsave
from miniSMLM.utils import Tracker2D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

config_path = 'run_track_2d.json'
with open(config_path, 'r') as f:
    config = json.load(f)

prefixes = [
'230923_j646_5pM_1.5hours_200frames_10mW_live__1',
'230923_j646_5pM_1.5hours_200frames_10mW_live__8',
'230923_j646_5pM_1.5hours_200frames_10mW_live__9',
'230929_Hela_H2B_1000ng_8h_200frames_10mW_100ms_J646_10PM_1.25hours_fixed__10',
'230929_Hela_H2B_1000ng_8h_200frames_10mW_100ms_J646_10PM_1.25hours_fixed__11',
'230929_Hela_H2B_1000ng_8h_200frames_10mW_100ms_J646_10PM_1.25hours_fixed__8'
]

for prefix in prefixes:
    print("Processing " + prefix)
    path = config['analpath'] + prefix + '/' + prefix + '_spots.csv'
    savepath = '-tracked.csv'.join(path.split('.csv'))
    tracker = Tracker2D(config)
    spots = pd.read_csv(path)
    linked = tracker.link(spots,filter=True)
    imsd = tracker.imsd(linked)
    imsd.to_csv(savepath)
