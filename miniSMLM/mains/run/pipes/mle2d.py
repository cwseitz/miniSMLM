import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from miniSMLM.localize import LoGDetector
from miniSMLM.psf.psf2d import MLE2D_BFGS
import concurrent.futures

class PipelineMLE2D:
    """A collection of functions for maximum likelihood localization"""
    def __init__(self, config, dataset):
        self.config = config
        self.analpath = config['analpath']
        self.datapath = config['datapath']
        self.dataset = dataset
        self.stack = dataset.stack
        Path(self.analpath + self.dataset.name).mkdir(parents=True, exist_ok=True)
        self.cmos_params = [config['eta'], config['texp'], config['gain'],
                            config['offset'], config['var']]
        self.dump_config()

    def dump_config(self):
        with open(self.analpath + self.dataset.name + '/' + 'config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)

    def localize(self, plot_spots=False, plot_fit=False, tmax=None, max_workers=4):
        path = self.analpath + self.dataset.name + '/' + self.dataset.name + '_spots.csv'
        file = Path(path)
        nt, nx, ny = self.stack.shape
        if tmax is not None:
            nt = tmax
        threshold = self.config['thresh_log']
        spotst = []
        if not file.exists():
            # Parallelize over frames with a limit on max_workers
            frames = [(n, self.stack[n], threshold, plot_spots, plot_fit) for n in range(nt)]
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self.process_frame, frames))
            spotst = pd.concat(results)
            
            self.save(spotst)
        else:
            print('Spot files exist. Skipping')
            spotst = pd.read_csv(path)
        return spotst

    def process_frame(self, args):
        n, framed, threshold, plot_spots, plot_fit = args
        print(f'Detecting in frame {n}')
        log = LoGDetector(framed, threshold=threshold)
        spots = log.detect()  # Image coordinates
        if plot_spots:
            log.show()
            plt.show()
        spots = self.fit(framed, spots, plot_fit=plot_fit)
        spots = spots.assign(frame=n)
        return spots

    def fit(self, frame, spots, plot_fit=False, max_workers=4):
        config = self.config
        patchw = config['patchw']
        # Prepare inputs for parallel processing
        spot_indices = spots.index.tolist()
        spot_coords = spots[['x', 'y']].values.tolist()
        args = [(i, x0, y0, frame, patchw, plot_fit) for i, (x0, y0) in zip(spot_indices, spot_coords)]
        # Parallelize fitting with a limit on max_workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.fit_spot, args))
        for i, result in zip(spot_indices, results):
            x_mle, y_mle, N0, conv = result
            spots.at[i, 'x_mle'] = x_mle
            spots.at[i, 'y_mle'] = y_mle
            spots.at[i, 'N0'] = N0
            spots.at[i, 'conv'] = conv
        return spots

    def fit_spot(self, args):
        i, x0, y0, frame, patchw, plot_fit = args
        start = time.time()
        x0 = int(x0)
        y0 = int(y0)
        adu = frame[x0 - patchw:x0 + patchw + 1, y0 - patchw:y0 + patchw + 1]
        adu = adu - self.cmos_params[3]
        adu = np.clip(adu, 0, None)
        theta0 = np.array([patchw, patchw, self.config['N0']])
        opt = MLE2D_BFGS(theta0, adu, self.config)  # Cartesian coordinates with top-left origin
        theta_mle, loglike, conv, err = opt.optimize(max_iters=self.config['max_iters'],
                                                     plot_fit=plot_fit)
        dx = theta_mle[1] - patchw
        dy = theta_mle[0] - patchw
        x_mle = x0 + dx  # Switch back to image coordinates
        y_mle = y0 + dy
        N0 = theta_mle[2]
        end = time.time()
        elapsed = end - start
        print(f'Fit spot {i} in {elapsed:.2f} sec')
        return x_mle, y_mle, N0, conv

    def save(self, spotst):
        path = self.analpath + self.dataset.name + '/' + self.dataset.name + '_spots.csv'
        spotst.to_csv(path, index=False)