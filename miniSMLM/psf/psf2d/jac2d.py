import numpy as np
from .psf2d import *
from .ill2d_auto import *

def jaciso2d(theta,adu,sigma,cam_params):
    nx,ny = adu.shape
    ntheta = len(theta)
    x0,y0,N0 = theta
    X,Y = np.meshgrid(np.arange(0,nx),np.arange(0,ny),indexing='ij')
    J1 = jac1(X,Y,theta,sigma,cam_params)
    J1 = J1.reshape((ntheta,nx**2))
    J2 = jac2(adu,X,Y,theta,sigma,cam_params)
    J = J1 @ J2
    return J
    
"""
def jaciso_auto2d(theta,adu,cmos_params):
    nx,ny,eta,texp,gain,offset,var = cmos_params
    ntheta = len(theta)
    theta = theta.reshape((ntheta,))
    ill = isologlike_auto2d(adu,eta,texp,gain,var)
    jacobian_ = jacobian(ill)
    jac = jacobian_(theta)
    return jac
"""
