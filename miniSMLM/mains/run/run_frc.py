import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from miniSMLM.utils import FRC, KDE

config_path = 'run_frc.json'
with open(config_path, 'r') as f:
    config = json.load(f)

prefixes = ['240109_wt_JF646_3pm_overnight_30mW_10ms___17']

for prefix in prefixes:
    print("Processing " + prefix)
    spots = pd.read_csv(config['path']+prefix+'.csv')
    spots['x [nm]'] /= 108.3
    spots['y [nm]'] /= 108.3

    
    kde_ = KDE(spots)
    kde = kde_.get_kde(pos=['x [nm]','y [nm]'])

    frc1 = FRC(spots)
    result1 = frc1.compute_frc(nsamples=5000,plot_kde=False,window_hw=500)
    xs_nm_freq1, frc_curve1 = result1

    frc2 = FRC(spots)
    result2 = frc2.compute_frc(nsamples=100000,plot_kde=False,window_hw=500)
    xs_nm_freq2, frc_curve2 = result2
    
    frc3 = FRC(spots)
    result3 = frc3.compute_frc(nsamples=200000,plot_kde=False,window_hw=500)
    xs_nm_freq3, frc_curve3 = result3
    
    fig,ax=plt.subplots(1,2,figsize=(8,4))
    ax[0].imshow(kde,cmap='gray',vmin=0,vmax=1.0)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_xlim([800,1400]); ax[0].set_ylim([900,1500])
    ax[1].scatter(1e3*xs_nm_freq1,frc_curve1,label='1%',
                  facecolor='white',edgecolor='black')
    ax[1].scatter(1e3*xs_nm_freq2,frc_curve2,label='20%',
                  facecolor='white',edgecolor='red')
    ax[1].scatter(1e3*xs_nm_freq3,frc_curve3,label='40%',
                  facecolor='white',edgecolor='blue')
    ax[1].set_xlabel(r'Spatial Frequency ($\mathrm{um}^{-1}$)')
    ax[1].hlines(1/7,0,50,color='red',linestyle='--')
    ax[1].set_ylabel('FRC')
    ax[1].set_xlim([0,30])
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].legend(frameon=False)
    plt.tight_layout()
    plt.savefig('/home/cwseitz/Desktop/FRC.png',dpi=300)
    plt.show()
    
