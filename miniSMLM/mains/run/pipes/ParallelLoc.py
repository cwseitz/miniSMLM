import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from miniSMLM.localize import LoGDetector
from miniSMLM.psf.psf2d import MLE2D, MLE2D_MCMC
from multiprocess import Pool

class Localizer:
    def __init__(self, n, pipe):
        self.n = n
        self.pipe = pipe

    def localize(self,plot_spots=False,plot_fit=False,tmax=None):
        path = self.pipe.analpath+self.pipe.dataset.name+'/'+self.pipe.dataset.name+'_spots.csv'
        file = Path(path)
        nx,ny = self.pipe.stack[self.n].shape
        framespots = []
        if not file.exists():
            print(f'Det in frame {self.n+1}')
            frame = self.pipe.stack[self.n]
            log = LoGDetector(frame,threshold=self.pipe.config['thresh_log'])
            framespots = log.detect() #image coordinates
            if plot_spots:
                log.show(); plt.show()
            framespots = self.fit(frame,framespots,plot_fit=plot_fit)
            framespots = framespots.assign(frame=self.n)
        else:
            print('Spot files exist. Skipping')
        return framespots

    def fit(self,frame,spots,plot_fit=False): #need to unharcode tol on line 49
        config = self.pipe.config
        patchw = self.pipe.config['patchw']
        for i in spots.index:
            # start = time.time()
            x0 = int(spots.at[i,'x'])
            y0 = int(spots.at[i,'y'])
            adu = frame[x0-patchw:x0+patchw+1,y0-patchw:y0+patchw+1]
            adu = adu - self.pipe.cmos_params[3]
            adu = np.clip(adu,0,None)
            theta0 = np.array([patchw,patchw,config['sigma'],config['N0']])
            opt = MLE2D(theta0,adu,config) #cartesian coordinates with top-left origin
            theta_mle, loglike, conv = opt.optimize(max_iters=config['max_iters'],
                                                         plot_fit=plot_fit,
                                                         tol=1e-4, #come back to this and un-hardcode it
                                                         lr=self.pipe.lr)
            dx = theta_mle[1] - patchw; dy = theta_mle[0] - patchw
            spots.at[i, 'x_mle'] = x0 + dx #switch back to image coordinates
            spots.at[i, 'y_mle'] = y0 + dy
            spots.at[i, 'N0'] = theta_mle[2]
            spots.at[i, 'conv'] = conv
            # end = time.time()
            # elapsed = end-start
            # print(f'Fit spot {i} in {elapsed} sec')
        return spots