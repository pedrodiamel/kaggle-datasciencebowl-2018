

import os
import sys
import numpy as np
import PIL.Image
import scipy.misc

import cv2
import skimage.color as skcolor
import skimage.util as skutl
import random
import csv
import h5py

from skimage import io, transform, morphology, filters
from scipy import ndimage
import skimage.morphology as morph
import skfmm

#tranformations 

def isrgb( image ):
    return len(image.shape)==2 or (image.shape==3 and image.shape[2]==1)

def to_rgb( image ):
    #to rgb
    if isrgb( image ):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def to_gray( image ):
    if not isrgb( image ):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def to_channels( image, ch ):
    if ch == 1:
        image = to_gray( image )[:,:,np.newaxis]
    elif ch == 3:
        image = to_rgb( image )
    else:
        assert(False)
    return image

def summary(data):
    print(data.shape, data.min(), data.max())

def ffftshift2(h):    
    H = np.fft.fft2(h)
    H = np.abs( np.fft.fftshift( H ) )
    return H

def to_one_hot( x, nc ):
    y = np.zeros((nc)); y[x] = 1.0
    return y

def norm_fro(a,b): 
    return np.sum( (a-b)**2.0 );

def complex2vector(c):
    '''complex to vector'''    
    return np.concatenate( ( c.real, c.imag ) , axis=1 )