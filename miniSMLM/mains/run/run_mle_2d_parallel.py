from pipes import Localizer
from miniSMLM.utils import SMLMDataset
from miniSMLM.utils import KDE, make_animation, Filter
from skimage.io import imsave
import matplotlib.pyplot as plt
import pandas as pd
import json
from multiprocessing import Pool
from PIL import Image, ImageEnhance

config_path = 'miniSMLM-main/miniSMLM/mains/run/run_mle_2d_parallel.json' 
prefixes = [
            '240404_Control__20_MMStack_Default.ome', 
            ]
show_image = True 
auto_thresh = True 
with open(config_path, 'r') as f:
    config = json.load(f)

# thresholding framework
if auto_thresh:
    threshs = []
    for prefix in prefixes:
        dataset = SMLMDataset(config['datapath']+prefix,prefix)
        threshs.append(dataset.defineThresh(numFrames = 10)) # numFrames should be even maybe
    print(threshs)


# helper function for multiprocessing'''
def job(n):
    frame = Localizer(n, config, dataset)
    spots = frame.localize()
    return spots

# main for all images
for prefix in prefixes:
    # set up imagee variables
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix) 
    config['thresh_log'] = threshs[0] 
    threshs.pop(0) 
    
    # call to helper function for parallelization
    frames = range(min(config['tmax'], len(dataset.stack))) 
    outputs = []
    if __name__ == '__main__': 
        with Pool(config['processes']) as p: 
            outputs = p.map(job, frames) 
    Localizer.save(outputs, prefix, config['analpath'])


    # filter spots and resave localizations for cluster analysis
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    spots = Filter(spots).filter()
    spots.to_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv', index = False)
    
    # create and save image
    make_animation(dataset.stack,spots)
    render = KDE(spots).forward(sigma=2.0)
    print(f"KDE complete on {prefix}")
    imsave(config['analpath']+prefix+'/'+prefix+'-kde.tif',render)
    if show_image:
        plt.imshow(render, cmap = 'gray')
        plt.show()
