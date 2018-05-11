

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






class ElasticTensorDistort(object):
    '''
    Elastic transformation with torch 
    '''

    def __init__(self, size_grid=50, deform=15, prob=0.5): 
        self.size_grid = size_grid
        self.deform = deform
        self.prob = prob

    def __call__(self, sample):
        
        image, label, weight = sample['image'], sample['label'], sample['weight']
        width, height = image.size(1),image.size(2)   

        if random.random() < self.prob:

            # get transform
            grid = torch_elastic_transform( (height, width) , self.size_grid, self.deform )

            # apply tranform
            image_t  = grid_sample(torch.unsqueeze(image,dim=0), grid).data[0,...]
            label_t  = grid_sample(torch.unsqueeze(label.float(),dim=0), grid).round().data[0,...]
            weight_t = grid_sample(torch.unsqueeze(weight,dim=0), grid).data[0,...]

            if label_t[1,...].sum() > 0:
                image = image_t;
                label = label_t;
                weight = weight_t;
                    
        return {'image': image, 'label': label, 'weight': weight}







## color ====================================================================================


class RandomSaturation:
    def __init__(self, limit=0.3, prob=0.5):
        self.limit = limit
        self.prob = prob
    def __call__(self, img):
        # dont work :(
        if random.random() < self.prob:
            
            alpha = 1.0 + random.uniform(-self.limit, self.limit)
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype( np.float32 )
            img[..., :3] = alpha * img[..., :3].astype( np.float32 ) + (1.0 - alpha) * gray
            img[..., :3] = clip(img[..., :3], dtype, maxval)

        return img


class Grayscale(object):    
    def __init__(self, prob=.5):
        self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            grayimage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(grayimage, cv2.COLOR_GRAY2RGB)
        return img


# https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
# https://github.com/fchollet/keras/pull/4806/files
# https://zhuanlan.zhihu.com/p/24425116
# http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html

class RandomHueSaturationShift(object):
    
    def __init__(self, limit=0.3, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, image):
        
        if random.random() < self.prob:
            
            alpha = 1.0 + random.uniform(-self.limit, self.limit)
            h   = int(alpha*180)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 170
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return image       
    


class RandomHueSaturationValue(object):
        
    def __init__(self, hue_shift_limit=(-5, 5), sat_shift_limit=(-11, 11), val_shift_limit=(-11, 11), prob=0.5):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(image)
            hue_shift = random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1])
            h = cv2.add(h, hue_shift)
            sat_shift = random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        return image


# https://www.kaggle.com/c/data-science-bowl-2018/discussion/53940
class ColorShift(object):    
    def __init__(self, r_shift_limit=(-128, 128), g_shift_limit=(-128, 128), b_shift_limit=(-128, 128), prob=0.5):
        self.r_shift_limit = r_shift_limit
        self.g_shift_limit = g_shift_limit
        self.b_shift_limit = b_shift_limit
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            r,g,b = cv2.split(image)
            r_shift = random.uniform(self.r_shift_limit[0], self.r_shift_limit[1])
            r = cv2.add(r, r_shift)
            g_shift = random.uniform(self.g_shift_limit[0], self.g_shift_limit[1])
            g = cv2.add(g, g_shift)
            b_shift = random.uniform(self.b_shift_limit[0], self.b_shift_limit[1])
            b = cv2.add(b, b_shift)
            image = cv2.merge((r, g, b))
        return image

class Negative(object):    
    def __init__(self, prob=.5):
        self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            img = 255-img
        return img

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