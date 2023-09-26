import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from SMLM.utils import KDE
from tifffile import imread
from skimage.io import imsave

with open('run_kde_2d.json', 'r') as f:
    config = json.load(f)

prefixes = [
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_11',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_19',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_21',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_25',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_28',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_29',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_5',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_7',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_8',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_9'
]


prefixes += [
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_11',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_12',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_13',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_14',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_15',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_16',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_17',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_20',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_6',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_9'

]

prefixes += [
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_12',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_13',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_17',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_18',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_1',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_2',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_3',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_5',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_7',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_8'
]

plot=False
for n,prefix in enumerate(prefixes):
    print("Processing " + prefix)
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    spots = spots.loc[(spots['N0'] > 10) & (spots['N0'] < 5000) & (spots['x_mle'] > 0) & (spots['y_mle'] > 0)]
    spots = spots.sample(10000)
    render = KDE(spots).get_kde(sigma=2.0)
    
    if plot:
        fig,ax=plt.subplots()
        ax.imshow(render,cmap='gray',vmin=0,vmax=0.2)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

        fig,ax=plt.subplots()
        ax.scatter(spots['y_mle'],spots['x_mle'],s=1,color='black',marker='x')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

        plt.show()
    
    imsave(prefix+'-kde.tif',render)
