import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import blob_log
from skimage.util import img_as_float

class LoGDetector:
    def __init__(self,X,min_sigma=1,max_sigma=3,num_sigma=5,threshold=0.5,
                 overlap=0.5,show_scalebar=True,pixel_size=108.3,
                 blob_marker='x',patchw=3,
                 blob_markersize=10,blob_markercolor=(0,0,1,0.8)):

        self.X = X
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.overlap = overlap
        self.patchw = patchw
        self.show_scalebar = show_scalebar
        self.pixel_size = pixel_size
        self.blob_marker = blob_marker
        self.blob_markersize = blob_markersize
        self.blob_markercolor = blob_markercolor

    def detect(self):

        blobs = blob_log(img_as_float(self.X),
                         min_sigma=self.min_sigma,
                         max_sigma=self.max_sigma,
                         num_sigma=self.num_sigma,
                         threshold=self.threshold,
                         overlap=self.overlap,
                         exclude_border=5
                         )

        columns = ['x', 'y', 'peak']
        self.spots = pd.DataFrame([], columns=columns)
        self.spots['x'] = blobs[:,0]
        self.spots['y'] = blobs[:,1]

        for i in self.spots.index:
            x = int(self.spots.at[i, 'x'])
            y = int(self.spots.at[i, 'y'])
            blob = self.X[x-self.patchw:x+self.patchw, y-self.patchw:y+self.patchw]
            self.spots.at[i, 'peak'] = blob.max()

        return self.spots

    def show(self,X=None,ax=None):

       if ax is None:
           fig, ax = plt.subplots(figsize=(6,6))
       if X is None:
           X = self.X
       ax.imshow(X, cmap="gray", aspect='equal')
       ax.scatter(self.spots['y'],self.spots['x'],color='red',marker='x')
    #    if self.show_scalebar:
    #        font = {'family': 'arial', 'weight': 'bold','size': 16}
    #        scalebar = ScaleBar(self.pixel_size, 'nm', location = 'upper right',
    #            font_properties=font, box_color = 'black', color='white')
    #        scalebar.length_fraction = .3
    #        scalebar.height_fraction = .025
    #        ax.add_artist(scalebar)
