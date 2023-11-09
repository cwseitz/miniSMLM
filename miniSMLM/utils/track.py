import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import trackpy as tp
import json

class Tracker2D:
    def __init__(self,config,prefix):
        self.config = config       
    def link(self,spots,search_range=3,memory=5,filter=False,min_length=10):  
        config = self.config         
        spots = spots.dropna(subset=['x','y','frame'])
        if filter:
            spots = tp.link_df(spots,search_range=search_range,memory=memory)
            spots = tp.filter_stubs(spots,config['min_length'])
            spots = spots.reset_index(drop=True)
        else:
            spots = tp.link_df(spots,search_range=config['search_range'],memory=config['memory'])
            spots = spots.reset_index(drop=True)
        return spots
        
    def imsd(self,spots):
        config = self.config    
        return tp.imsd(spots,config['mpp'],config['fps'],
                       max_lagtime=config['max_lagtime'],
                       statistic='msd',pos_columns=['x_mle','y_mle'])

        
        
