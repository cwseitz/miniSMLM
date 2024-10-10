import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from miniSMLM.localize import LoGDetector
from miniSMLM.psf.psf2d import MLE2D_BFGS
from numpy.linalg import inv

class PipelineMLE2D:
    """A collection of functions for maximum likelihood localization"""
    def __init__(self,config,dataset):
        self.config = config
        self.analpath = config['analpath']
        self.datapath = config['datapath']
        self.dataset = dataset
        self.stack = dataset.stack
        Path(self.analpath+self.dataset.name).mkdir(parents=True, exist_ok=True)
        self.cmos_params = [config['eta'],config['texp'],config['gain'],
                            config['offset'],config['var']]
        self.dump_config()
    def dump_config(self):
        with open(self.analpath+self.dataset.name+'/'+'config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
    def localize(self,plot_spots=False,plot_fit=False,tmax=None):
        path = self.analpath+self.dataset.name+'/'+self.dataset.name+'_spots.csv'
        file = Path(path)
        nt,nx,ny = self.stack.shape
        if tmax is not None: nt = tmax
        threshold = self.config['thresh_log']
        spotst = []
        if not file.exists():
            for n in range(nt):
                print(f'Det in frame {n}')
                framed = self.stack[n]
                log = LoGDetector(framed,threshold=threshold)
                spots = log.detect() #image coordinates
                if plot_spots:
                    print(spots)
                    log.show(); plt.show()
                spots = self.fit(framed,spots,plot_fit=plot_fit)
                spots = spots.assign(frame=n)
                spotst.append(spots)
            spotst = pd.concat(spotst)
            self.save(spotst)
        else:
            print('Spot files exist. Skipping')
        return spotst

    def fit(self,frame,spots,plot_fit=False):
        config = self.config
        patchw = self.config['patchw']
        for i in spots.index:
            start = time.time()
            x0 = int(spots.at[i,'x']) #image coordinates (row)
            y0 = int(spots.at[i,'y']) #image coordinates (column)
            adu = frame[x0-patchw:x0+patchw+1,y0-patchw:y0+patchw+1]
            adu = adu - self.cmos_params[3]
            adu = np.clip(adu,0,None)
            theta0 = np.array([patchw,patchw,self.config['N0']])
            opt = MLE2D_BFGS(theta0,adu,self.config) #cartesian coordinates with top-left origin
            theta_mle, loglike, conv, err = opt.optimize(max_iters=config['max_iters'],
                                                         plot_fit=plot_fit)
            #theta_mle is in subpixel image coordinates (row,column) within the patch
            dx = theta_mle[0] - patchw; dy = theta_mle[1] - patchw
            spots.at[i, 'x_mle'] = x0 + dx
            spots.at[i, 'y_mle'] = y0 + dy
            spots.at[i, 'N0'] = theta_mle[2]
            spots.at[i, 'conv'] = conv
            end = time.time()
            elapsed = end-start
            print(f'Fit spot {i} in {elapsed} sec')

        return spots
    def save(self,spotst):
        path = self.analpath+self.dataset.name+'/'+self.dataset.name+'_spots.csv'
        spotst.to_csv(path)
