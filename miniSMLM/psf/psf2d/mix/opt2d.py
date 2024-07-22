import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import sys
import os
import seaborn as sns
import pandas as pd

from ..psf2d import *
from .mll2d import *
from .jac2d import *
from multiprocessing import Pool
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

"""
Optimizers for multi-emitter fitting
"""

class MixMCMC:
    def __init__(self,theta0,adu,config):
        self.theta0 = theta0
        self.adu = adu
        self.sigma = config['sigma']
        self.N0 = config['N0']
        self.cam_params = [config['eta'],config['texp'],config['gain'],
                           config['offset'],config['var']]

    def log_prior(self, theta):
        if np.all(theta >= 2.0) and np.all(theta <= 8.0):
            return 0.0
        return -np.inf

    def log_likelihood(self,theta):
        try:
            log_like = -1*mixloglike(theta,self.adu,self.sigma,self.N0,self.cam_params)
            if np.isnan(log_like) or np.isinf(log_like):
                return -np.inf
            return log_like
        except OverflowError:
            return -np.inf

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def run_mcmc(self, nwalkers=100, nsteps=1000, plot_fit=False):
        ndim = self.theta0.size
        pos = self.theta0 + 1e-3 * np.random.randn(nwalkers, ndim)

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, pool=pool)
            sampler.run_mcmc(pos, nsteps, progress=True)

        samples = sampler.get_chain(discard=100, thin=5, flat=True)
        if plot_fit:
            self.plot_fit(samples)
        return samples
        
    def cluster_samples(self,samples,N):
        kmeans=KMeans(n_clusters=N,n_init='auto').fit(samples)
        theta_est = kmeans.cluster_centers_
        return theta_est

    def plot_fit(self,samples,adu,theta_true,N):
        kmeans=KMeans(n_clusters=N).fit(samples)
        labels=kmeans.labels_
        fig,ax=plt.subplots(figsize=(3,3))
        ax.invert_yaxis()
        ax.scatter(samples[:,1],samples[:,0],c=labels,marker='x',s=1)
        ax.scatter(theta_true[1,:],theta_true[0,:],
                   marker='x',color='red',s=20)
        ax.set_xlabel(r'$x_0$')
        ax.set_ylabel(r'$y_0$')
        ax.spines[['right','top']].set_visible(False)
        ax_inset = inset_axes(ax,width="40%",
                              height="40%",loc='upper right')
        ax_inset.imshow(adu,cmap='gray')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        plt.tight_layout()
        plt.savefig('/home/cwseitz/Desktop/Samples.png',dpi=300)
        plt.show()



