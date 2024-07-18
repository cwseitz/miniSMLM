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
from sklearn.mixture import BayesianGaussianMixture
from multiprocessing import Pool
from scipy.optimize import minimize

"""
Optimizers for multi-emitter fitting
"""

class MixMCMCParallel:
    def __init__(self, theta0,adu,config):
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

    def plot_fit(self, samples):
        fig = corner.corner(samples, labels=["param" + str(i) for i in range(samples.shape[1])])
        fig.show()
        plt.show()

    def find_modes_dpgmm(self, samples, max_components=10):
        dpgmm = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=1e-3,
            random_state=0
        )
        samples += np.random.normal(0, 0.01, size=samples.shape)
        dpgmm.fit(samples)
        modes = dpgmm.means_
        labels = dpgmm.predict(samples)
        #self.plot_dpgmm_fit(samples, dpgmm, labels)
        return modes

    def plot_dpgmm_fit(self, samples, dpgmm, labels):
        df = pd.DataFrame(samples, columns=[r'$x_0$', r'$y_0$'])
        df['cluster'] = labels
        sns.set_theme(font_scale=1.5, style='ticks')
        sns.pairplot(df, hue='cluster', diag_kind='kde', palette='tab10')
        plt.show()


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
    
    def run_mcmc(self,nwalkers=100,nsteps=1000,plot_fit=False):
        ndim = self.theta0.size
        pos = self.theta0 + 1e-3*np.random.randn(nwalkers,ndim)
        sampler = emcee.EnsembleSampler(nwalkers,ndim,self.log_probability)
        sampler.run_mcmc(pos,nsteps,progress=True)
        samples = sampler.get_chain(discard=100,thin=5,flat=True)
        if plot_fit:
            self.plot_fit(samples)
        return samples
    
    def plot_fit(self,samples):
        fig = corner.corner(samples, labels=["param" + str(i) for i in range(samples.shape[1])])
        fig.show()
        plt.show()
        
    def find_modes_dpgmm(self, samples, max_components=10):
        dpgmm = BayesianGaussianMixture(n_components=max_components, covariance_type='full', weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=1e-3, random_state=0)
        samples += np.random.normal(0,0.01,size=samples.shape)
        dpgmm.fit(samples)
        modes = dpgmm.means_
        labels = dpgmm.predict(samples)

        print("Modes found by DPGMM:")
        print(modes)

        self.plot_dpgmm_fit(samples, dpgmm, labels)
        return modes

    def plot_dpgmm_fit(self, samples, dpgmm, labels):
        df = pd.DataFrame(samples, columns=[r'$x_0$',r'$y_0$'])
        df['cluster'] = labels
        sns.set_theme(font_scale=1.5,style='ticks')
        sns.pairplot(df,hue='cluster',diag_kind='kde', palette='tab10')
        plt.show()



