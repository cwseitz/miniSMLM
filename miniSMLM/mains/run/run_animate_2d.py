import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from miniSMLM.utils import make_animation
from tifffile import imread

with open('run_animate_2d.json', 'r') as f:
    config = json.load(f)

prefixes = []

for n,prefix in enumerate(prefixes):
    print("Processing " + prefix)
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    stack = imread(config['datapath']+prefix+'.tif')
    make_animation(stack,spots)
    plt.show()
