import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from miniSMLM.localize import LoGDetector
from pipes import TwoStatePoissonHMM


prefixes = [
'240110_Control_JF646_4pm_overnight_L640_30mW_10ms____10'
]

with open('run_lifetime.json', 'r') as f:
    config = json.load(f)

for prefix in prefixes:
    stack = imread(config['path'] + prefix + '.tif')
    pipe = TwoStatePoissonHMM(stack)
    pipe.forward(show_spots=False,show_hmm=False,show_hist=True,threshold=0.001)


