import numpy as np
import tifffile
import torch
from skimage.io import imread
from torch.utils.data import Dataset
from glob import glob

class SMLMDataset(Dataset):
    def __init__(self,path,name):
        self.stack = tifffile.imread(path+'.tif')
        self.name = name
        try:
            prefix = path.split('.')[0]
            npz = np.load(prefix+'.npz',allow_pickle=True)
            self.theta = npz['theta']
        except Exception as e:
            self.theta = None
    def get_theta(self,idx):
        if self.theta is not None:
            return self.theta[idx]
        else:
            return None
    def __len__(self):
        nb,nc,nx,ny = self.stack.shape
        return nb
    def __getitem__(self, idx):
        adu = self.stack[idx]
        return adu
        
    def defineThresh(self, sigma=0.92, numFrames = 100):
            trial_thresh = float(input('Threshold to try: '))
            check = trial_thresh
            interval = int(len(self.stack)/numFrames)
    
            # run loop until user inputs that the threshold is satisfactory
            while (check != '0'):
                fig, axs = plt.subplots(nrows = 2, ncols = int(numFrames/2), figsize = (10,6))
                trial_thresh = float(check)
                logs = []
                for i in range(numFrames):
                    log = LoGDetector(self.stack[interval*i], threshold=trial_thresh)
                    log.detect()
    
                    # plots the detected spots
                    x_plt = int((2*i)/numFrames)
                    y_plt = int(((2*i) % numFrames)/2)
                    axs[x_plt, y_plt].set_title(f'Frame {interval*i}, {len(log.spots)} spots', )
                    axs[x_plt, y_plt].imshow(self.stack[interval*i], cmap="gray", aspect='equal')
                    axs[x_plt, y_plt].scatter(log.spots['y'], log.spots['x'], color = 'red', marker = 'x')
    
                plt.show()
                
                check = input("Enter 0 if acceptable. Enter a new threshold if not: ")
            
            return trial_thresh
