import os
import numpy as np
from crowdsource import psf
from crowdsource import decam_proc
from crowdsource import crowdsource_base

from astropy.io import fits

from astropy.visualization import ZScaleInterval
import pandas as pd
from astropy.wcs import WCS
from astropy.io import fits
from astropy.wcs import utils

import matplotlib.pyplot as plt
import astropy.wcs as wcs

import psycopg2
import psycopg2.extras
import re

path = "cfs/cdirs/m937/www/decat/2022A-724693/220420/c4d_220420_235947_ori"

def get_hduls(img_fp, mask_fp, weight_fp, Plot = True):
    """
    Get fits hduls from the filepaths, and plot the images if wanted.
    """
    img_hdul = fits.open(img_fp)
    mask_hdul = fits.open(mask_fp)
    weight_hdul = fits.open(weight_fp)
    
    if Plot == True:

        plt.figure()
        vmin, vmax = ZScaleInterval().get_limits(img_hdul[1].data)
        plt.imshow(img_hdul[1].data, cmap = "gray", vmin=vmin, vmax=vmax)
        plt.colorbar()

        plt.figure()
        vmin, vmax = ZScaleInterval().get_limits(mask_hdul[1].data)
        plt.imshow(mask_hdul[1].data, cmap = "gray", vmin=vmin, vmax=vmax)
        plt.colorbar()

        plt.figure()
        vmin, vmax = ZScaleInterval().get_limits(weight_hdul[1].data)
        plt.imshow(weight_hdul[1].data, cmap = "gray", vmin=vmin, vmax=vmax)
        plt.colorbar()
        
    return img_hdul, mask_hdul, weight_hdul


def get_pars(img, weight, mask = None, fwhm=4, miniter = 4, maxiter = 10):
    """
    Do the non-forced image fit. This is done using a gaussian psf (fix this) 
    instead of a moffat psf.
    """
    
    #MAKE PSF OBJECT
    psf_ = psf.gaussian_psf(fwhm, deriv=False) # get the psf stamps
    psf_ = np.reshape(psf_, (19, 19)) # reshape the psf into the shape the SimplePSF object needs.
    psf__ = psf.SimplePSF(psf_) # make a SimplePSF psf object with the stamps.
    
    #CROWDSOURCE GET XS AND YS
    pars = crowdsource_base.fit_im(img, psf__, weight=weight, 
                               verbose=True, miniter=miniter, maxiter=maxiter, 
                               refit_psf=True, derivcentroids=True) # Get the crowdsource fit (including sky subtraction).
    
    
    return pars

def get_pars_force(img, weight, x, y, mask = None, fwhm=4, miniter = 4, maxiter = 10):
    """
    Get the forced image fit. use the moffat psf to allow for non-gaussian psfs, and refit_psf = True 
    to allow the psf to move a little bit.
    """
    
    #MAKE PSF OBJECT
    psf__ = psf.MoffatPSF(fwhm, beta=3)
    
    #CROWDSOURCE GET XS AND YS
    
    pars = crowdsource_base.fit_im_force(img, 
                                 x, y, 
                                 psf__, weight = weight, psfderiv = False, refit_psf=True) # Do the forced fit
     
    return pars




class utils:
    
    def get_weight(weighthdul, c):
        return np.flipud(weighthdul[1].data[c[2]:c[3], c[0]:c[1]]) # Gets flipped because of numpy indexing

    def get_full_pos(pars, s, c):
        ymin, ymax, xmin, xmax = c
        
        ypres = np.reshape(pars[0]["y"], (pars[0]["x"].shape[0], 1))
        xpres = np.reshape(pars[0]["x"], (pars[0]["x"].shape[0], 1))

        xs = xpres[xpres>0][ypres[xpres>0]>0]
        ys = ypres[xpres>0][ypres[xpres>0]>0]

        xs = np.reshape(xs, (xs.shape[0], 1))
        ys = np.reshape(ys, (xs.shape[0], 1))

        ypre = xs - s
        xpre = ys - s

        y = -1*ypre
        x = xpre

        ypre = y + s - 1
        xpre = x + s

        xref = xpre + ymin
        yref = ypre + xmin

        return xref, yref

    def get_radecs(imhdul, xref, yref):
        ws = wcs.WCS(imhdul[1].header)  # create a wcs object with the quadrant object's header
        return ws.wcs_pix2world(np.transpose(np.array([xref, yref+0.5]))[0], 0)  
        # return the Ra/Dec positions as a numpy array (ofset by 0.5 pixels to account for differences in numpy   
        # indexed image and full image xlim/ylim)


    def get_rel_pos(x, y, s, c):
        ymin, ymax, xmin, xmax = c
        # Get Relative Positions
        xref = x - ymin
        yref = y - xmin

        xref = xref - s
        yref = yref - s

        yref = -1*yref

        xref = xref + s
        yref = yref + s

        yref = yref-1

        return xref, yref

    def get_xys(imhdul, radecs):   
        ws = wcs.WCS(imhdul[1].header)  # create a wcs object with the quadrant object's header
        xys = ws.wcs_world2pix(radecs, 0)  # return the Ra/Dec positions as a numpy array

        x = np.transpose(xys)[0]
        y = np.transpose(xys)[1]

        return x, y

