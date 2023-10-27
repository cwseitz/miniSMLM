from pipes import PipelineMLE2D
from miniSMLM.torch.dataset import SMLMDataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

with open('run_mle_2d.json', 'r') as f:
    config = json.load(f)

prefixes = []

for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    pipe = PipelineMLE2D(config,dataset)
    pipe.localize(plot_spots=False,plot_fit=False,tmax=200,run_deconv=False)
