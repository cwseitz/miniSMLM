import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from miniSMLM.localize import LoGDetector
from miniSMLM.psf.psf2d import MLE2D


class Localizer:
    """A collection of functions for maximum likelihood localization"""
    def __init__(self, n, config, dataset):
        self.n = n
        self.config = config
        self.analpath = config['analpath']
        self.datapath = config['datapath']
        self.dataset = dataset
        self.stack = dataset.stack
        self.lr = self.config['lr']
        self.spotst = []
        self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]
        Path(self.analpath+self.dataset.name).mkdir(parents=True, exist_ok=True)
        self.dump_config()

    def dump_config(self):
        with open(self.analpath+self.dataset.name+'/'+'config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
    
    def localize(self,plot_spots=False,plot_fit=False):
        path = self.analpath+self.dataset.name+'/'+self.dataset.name+'_spots.csv'
        file = Path(path)
        nx,ny = self.stack[self.n].shape
        framespots = []
        if not file.exists():
            print(f'Det in frame {self.n+1}')
            frame = self.stack[self.n]
            log = LoGDetector(frame,threshold=self.config['thresh_log'])
            framespots = log.detect() #image coordinates
            if plot_spots:
                log.show(); plt.show()
            framespots = self.fit(frame,framespots,plot_fit=plot_fit)
            framespots = framespots.assign(frame=self.n)
        else:
            print('Spot files exist. Skipping')
        return framespots

    def fit(self,frame,spots,plot_fit=False): #need to unharcode tol on line 49
        patchw = self.config['patchw']
        for i in spots.index:
            # start = time.time()
            x0 = int(spots.at[i,'x'])
            y0 = int(spots.at[i,'y'])
            adu = frame[x0-patchw:x0+patchw+1,y0-patchw:y0+patchw+1]
            adu = adu - self.cmos_params[3]
            adu = np.clip(adu,0,None)
            theta0 = np.array([patchw,patchw,self.config['sigma'],self.config['N0']])
            opt = MLE2D(theta0,adu,self.config) #cartesian coordinates with top-left origin
            theta_mle, loglike, conv = opt.optimize(max_iters=self.config['max_iters'], #doesn't use BFGS
                                                         plot_fit=plot_fit,
                                                         tol=1e-4, #come back to this and un-hardcode it
                                                         lr=self.lr)
            dx = theta_mle[1] - patchw; dy = theta_mle[0] - patchw
            spots.at[i, 'x_mle'] = x0 + dx #switch back to image coordinates
            spots.at[i, 'y_mle'] = y0 + dy
            spots.at[i, 'N0'] = theta_mle[2]
            spots.at[i, 'conv'] = conv
            # end = time.time()
            # elapsed = end-start
            # print(f'Fit spot {i} in {elapsed} sec')
        return spots
    
    def save(mle_stack,prefix,folder):
        formatted = pd.DataFrame()
        for i in range(len(mle_stack)):
            formatted = pd.concat([formatted, mle_stack[i]]).drop_duplicates()
        
        print(formatted)

        path = folder+prefix+'/'+prefix+'_spots.csv'
        formatted.to_csv(path)

        return formatted
        
