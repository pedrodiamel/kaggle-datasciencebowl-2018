

import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

import numpy as np
import logging
import random as rnd
from collections import namedtuple
from skimage import io, transform
import cv2
import random

# Import stuff
from scipy.interpolate import griddata
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import color
from skimage import util as skutl
import scipy.misc
import math

import time
import itertools

from .grid_sample import grid_sample
from .tps_grid_gen import TPSGridGen
from . import utility as utl


import warnings
warnings.filterwarnings("ignore")

from . import utility as utl





## color ====================================================================================
# https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
# https://github.com/fchollet/keras/pull/4806/files
# https://zhuanlan.zhihu.com/p/24425116
# http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
  


# https://www.kaggle.com/c/data-science-bowl-2018/discussion/53940



class ColorPermutation(object):    
    def __init__(self, prob=.5):
        self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            indexs = [0,1,2]
            random.shuffle(indexs)
            img = img[:,:, indexs ]
        return img






class ColorDistort(object):
    '''
    Color distortion
    '''
    def __init__( self, tranforms=['brightness', 'gamma'] ):
        
        self.tranforms = tranforms
        self.dispach = {
            'brightness': RandomBrightness(),
            'hue_value': RandomHueSaturationValue(),
            'hue_shift': RandomHueSaturationShift(),
            'contrast': RandomContrast(),
            'gaussian_blur': RandomGaussianBlur(),
            'negative': Negative(prob=0.15),
            'gray': Grayscale(prob=0.15),
            'change_channel': ColorPermutation(),
            'gamma': RandomGamma(),
            'brightness_shift': RandomBrightnessShift(),        
            'clahe': CLAHE(),
        }

        assert( all(x in self.dispach.keys()  for x in tranforms ) )

    def __call__(self, sample):
        
        image = sample['image'] 
        for k in self.tranforms:
            image = self.dispach[k](image)    

        image  = cunsqueeze(image)    
        sample['image'] = image
        return sample



def to_mean_normalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.
    See ``Normalize`` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.
    Returns:
        Tensor: Normalized Tensor image.
    """

    # TODO: make efficient
    result_tensor = []
    for t, m, s in zip(tensor, mean, std):
        result_tensor.append(t.sub_(m).div_(s))
    return torch.stack(result_tensor, 0)

def to_white_normalize(tensor):
    
    new_tensor = []
    for t in tensor:
        t = t.sub_( t.min() )
        t = t.div_( t.max() )
        new_tensor.append( t )        
    return torch.stack(new_tensor, 0)
    

class Normalize(object):
    '''
    Color normalization
    '''

    def __init__( self, type='white' ):
        self.type = type

    def __call__(self, sample):
        
        image  = sample['image']
        if self.type == 'white':
            image = to_white_normalize(image)
        elif self.type == 'mean':
            image = to_mean_normalize(image)
        else: assert(False)
               
        sample['image'] = image
        return  sample