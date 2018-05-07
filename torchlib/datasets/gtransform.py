

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
from deep.datasets.grid_sample import grid_sample
from deep.datasets.tps_grid_gen import TPSGridGen


import warnings
warnings.filterwarnings("ignore")

from deep.datasets import utility as utl

def cunsqueeze(data):
    if len( data.shape ) == 2: 
        data = data[:,:,np.newaxis]
    return data   

def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)

def relabel( mask ):
    h, w = mask.shape
    relabel_dict = {}
    for i, k in enumerate(np.unique(mask)):
        if k == 0:
            relabel_dict[k] = 0
        else:
            relabel_dict[k] = i
    for i, j in product(range(h), range(w)):
        mask[i, j] = relabel_dict[mask[i, j]]
    return mask

def scale(image, x1, y1, x2, y2, dx, dy, size0x, size0y, sizex, sizey, limit, mode ): 
    y1 = dy; y2 = y1 + sizey
    x1 = dx; x2 = x1 + sizex
    image_pad = cv2.copyMakeBorder(image, limit, limit, limit, limit, borderType=cv2.BORDER_CONSTANT,value=[0,0,0])
    image_pad = cunsqueeze(image_pad)
    image_t = cv2.resize(image_pad[y1:y2, x1:x2, :], (size0x, size0y), interpolation=mode)
    image_t = cunsqueeze(image_t) 
    return image_t


# ELASTIC TRANSFORMATION
def elastic_transform(shape, size_grid, deform):
        
    m,n=shape[:2]
    grid_x, grid_y = np.mgrid[:m,:n]
    
    source = [] 
    destination = []

    for i in range(int(m/size_grid)+1):
        for j in range(int(n/size_grid)+1):            
            source = source + [np.array([i*size_grid, j*size_grid])]
            noisex = round(random.uniform(-deform,deform))
            noisey = round(random.uniform(-deform,deform))
            noise  = np.array( [noisex,noisey] )
            if i==0 or j==0 or i==int(m/size_grid) or j==int(n/size_grid): noise = np.array([0,0])
            destination = destination + [np.array([i*size_grid, j*size_grid])+noise ]

    source=np.vstack(source)
    destination=np.vstack(destination)
    destination[destination<0] = 0
    destination[destination>=n] = n-1
    
    grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')

    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(m,n)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(m,n)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')

    return map_x_32, map_y_32

def torch_elastic_transform(shape, size_grid, deform):
    
    target_height, target_width = shape[:2]
    target_control_points = torch.Tensor(list(itertools.product(
        torch.arange(-1.0, 1.00001, 2.0 / (size_grid-1)),
        torch.arange(-1.0, 1.00001, 2.0 / (size_grid-1)),
        )))

    source_control_points = target_control_points + torch.Tensor(target_control_points.size()).uniform_(-deform, deform)
    tps = TPSGridGen(target_height, target_width, target_control_points)
    source_coordinate = tps(Variable(torch.unsqueeze(source_control_points, 0)))
    grid = source_coordinate.view(1, target_height, target_width, 2)
    
    return grid

# GEOMETRICAL TRANSFORM
def geometric_transform( imsize, degree, translation, warp ):
    """
    Transform the image for data augmentation
    Arguments:
        * degree: Max rotation angle, in degrees. Direction of rotation is random.
        * translation: Max translation amount in both x and y directions,
            expressed as fraction of total image width/height
        * warp: Max warp amount for each of the 3 reference points,
            expressed as fraction of total image width/height

    Returns:
        * Transformed input as an np.array() object
    """

    height, width = imsize[:2]
    degree = degree * math.pi / 180

    # Rotation
    center = (width//2, height//2)
    theta = random.uniform(-degree, degree)
    rotation_mat = cv2.getRotationMatrix2D(center, -theta*180/math.pi, 1)
    
    # Translation
    x_offset = translation * width * random.uniform(-1, 1)
    y_offset = translation * height * random.uniform(-1, 1)
    translation_mat = np.float32( np.array([[1, 0, x_offset], [0, 1, y_offset]]) )

    # # Warp
    # # NOTE: The commented code below is left for reference
    # # The warp function tends to blur the image, so it is not useds
    
    src_triangle = np.float32([[0, 0], [0, height], [width, 0]])
    x_offsets = [warp * width * random.uniform(-1, 1) for _ in range(3)]
    y_offsets = [warp * height * random.uniform(-1, 1) for _ in range(3)]
    dst_triangle = np.float32([[x_offsets[0], y_offsets[0]],\
                             [x_offsets[1], height + y_offsets[1]],\
                             [width + x_offsets[2], y_offsets[2]]])
    warp_mat = cv2.getAffineTransform(src_triangle, dst_triangle)


    return rotation_mat, translation_mat, warp_mat 

# UNET RESIZE
def size_unet_transform(imagein, size=512, mode=cv2.INTER_CUBIC): 
    
    height, width, ch = imagein.shape;
    image = np.array(imagein.copy())
    
    asp = float(height)/width
    w = size
    h = int(w*asp)

    #resize mantaining aspect ratio
    #image_x = scipy.misc.imresize(image, (h,w), interp='bilinear', mode=mode)
    image_x = cv2.resize(image, (w,h) , interpolation = mode)
    
    # unzquese
    if len(image_x.shape) == 2:
        image_x = image_x[:,:,np.newaxis]
    image = np.zeros((w,w,ch))

    #crop image
    ini = int(round((w-h) / 2.0))
    image[ini:ini+h,:,:] = image_x

    #unet required input size
    downsampleFactor = 16;
    d4a_size   = 0;
    padInput   = (((d4a_size *2 +2 +2)*2 +2 +2)*2 +2 +2)*2 +2 +2;
    padOutput  = ((((d4a_size -2 -2)*2-2 -2)*2-2 -2)*2-2 -2)*2-2 -2;
    d4a_size   = math.ceil( (size - padOutput)/downsampleFactor);
    input_size = downsampleFactor*d4a_size + padInput;

    offset=(input_size-size)//2
    image_x = np.zeros((input_size,input_size, ch));

    #crop for required size
    image_x[ offset:-offset, offset:-offset, : ] = image
    
    return image_x

# RESIZE
class UnetResize(object):
    '''
    Unet resize 
    '''

    def __init__(self, imsize): 
        self.imsize = imsize

    def __call__(self, sample):
        
        image, label, weight = sample['image'], sample['label'], sample['weight']        
        # apply tranform
        image_t  = size_unet_transform(image,  self.imsize, mode=cv2.INTER_LINEAR )        
        label_t  = size_unet_transform(label,  self.imsize, mode=cv2.INTER_NEAREST )        
        weight_t = size_unet_transform(weight, self.imsize, mode=cv2.INTER_LINEAR )
        
        return {'image': image_t, 'label': label_t, 'weight': weight_t}

class RandomCrop(object):
    '''
    Random Crop
    '''

    def __init__(self, cropsize=(250,250) ): 
        self.cropsize = cropsize
        self.centercrop = CenterCrop(cropsize[0],cropsize[1])
    
    def __call__(self, sample):
        
        image, label, weight = sample['image'], sample['label'], sample['weight']
        cropsize = self.cropsize

        h,w = image.shape[:2]
        new_h, new_w = cropsize

        area = 0
        for i in range(10):            

            top  = random.randint( 0, h - new_h )
            left = random.randint( 0, w - new_w )

            image_t  = utl.imagecrop(  image, cropsize, top, left)
            label_t  = utl.imagecrop(  label, cropsize, top, left)
            weight_t = utl.imagecrop( weight, cropsize, top, left)

            area = label_t[:,:,1].sum()
            if area > 0:
                return { 'image': image_t, 'label': label_t, 'weight': weight_t }
                
        return self.centercrop(sample)


class CenterCrop(object):
    
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, sample):
        
        image, label, weight = sample['image'], sample['label'], sample['weight']
        
        h, w, c = image.shape
        dy = (h - self.height) // 2
        dx = (w - self.width)  // 2

        y1 = dy
        y2 = y1 + self.height
        x1 = dx
        x2 = x1 + self.width

        image_t  =  image[y1:y2, x1:x2, :]
        label_t  =  label[y1:y2, x1:x2, :]
        weight_t = weight[y1:y2, x1:x2, :]

        return {'image': image_t, 'label': label_t, 'weight': weight_t}



class ShiftScale(object):
    
    def __init__(self, limit=4, prob=.25):
        self.limit = limit
        self.prob = prob

    def __call__(self, sample):
        
        image, label, weight = sample['image'], sample['label'], sample['weight']

        limit = self.limit
        if random.random() < self.prob:
            
            height, width, channel = image.shape
            
            #assert(width == height)

            size0x = width
            size1x = width + 2*limit
            sizex = round(random.uniform(size0x, size1x))

            size0y = height
            size1y = height + 2*limit
            sizey = round(random.uniform(size0y, size1y))

            dx = round(random.uniform(0, size1x-sizex))
            dy = round(random.uniform(0, size1y-sizey))

            y1 = dy
            y2 = y1 + sizey
            x1 = dx
            x2 = x1 + sizex

            image_t  = scale(image,  x1, y1, x2, y2,  dx, dy, size0x, size0y, sizex, sizey, limit, cv2.INTER_LINEAR )
            label_t  = scale(label,  x1, y1, x2, y2,  dx, dy, size0x, size0y, sizex, sizey, limit, cv2.INTER_NEAREST )
            weight_t = scale(weight, x1, y1, x2, y2,  dx, dy, size0x, size0y, sizex, sizey, limit, cv2.INTER_LINEAR )

            area = label[:,:,1].sum()
            area_t = label_t[:,:,1].sum()
            if area_t/area > 0.50 and area_t > 50: #5x5x2 
                image  = image_t
                label  = label_t
                weight = weight_t        
            

        return {'image': image, 'label': label, 'weight': weight}


class RandomFlip(object):
    
    def __init__(self, prob=0.5 ): 
        self.prob = prob

    def __call__(self, sample):            
        image, label, weight = sample['image'], sample['label'], sample['weight']

        if random.random() < self.prob:    

            if random.random() < 0.5: 
                image, label, weight = [ cv2.flip(x,-1) for x in [image, label, weight] ]
            
            if random.random() < 0.5: 
                image, label, weight = [ cv2.flip(x, 1) for x in [image, label, weight] ]

            if random.random() < 0.5: 
                image, label, weight = [ cv2.flip(x, 0) for x in [image, label, weight] ]
       
            # op = random.randint(1,7)
            # if op==1: #rotate90
            #     image, label, weight = [ cv2.flip(x.transpose(1,0,2),1) for x in [image, label, weight] ]
            # if op==2: #rotate180 
            #     image, label, weight = [ cv2.flip(x,-1) for x in [image, label, weight] ]
            # if op==3: #rotate270
            #     image, label, weight = [ cv2.flip(x.transpose(1,0,2),0) for x in [image, label, weight] ]
            # if op==4: #flip left-right
            #     image, label, weight = [ cv2.flip(x,1) for x in [image, label, weight] ]
            # if op==5: #flip up-down
            #     image, label, weight = [ cv2.flip(x,0) for x in [image, label, weight] ]
            # if op==6:
            #     image, label, weight = [ cv2.flip(cv2.flip(x,1).transpose(1,0,2),1) for x in [image, label, weight] ]
            # if op==7:
            #     image, label, weight = [ cv2.flip(cv2.flip(x,0).transpose(1,0,2),1) for x in [image, label, weight] ]

            image  = cunsqueeze(image)
            label  = cunsqueeze(label)
            weight  = cunsqueeze(weight)

        return {'image': image, 'label': label, 'weight': weight}


class RandomHFlip(object):
    
    def __init__(self, prob=0.5 ): 
        self.prob = prob

    def __call__(self, sample):            
        image, label, weight = sample['image'], sample['label'], sample['weight']

        if random.random() < self.prob:    
            image, label, weight = [ cv2.flip(x,1) for x in [image, label, weight] ]
            image  = cunsqueeze(image)
            label  = cunsqueeze(label)
            weight  = cunsqueeze(weight)
        return {'image': image, 'label': label, 'weight': weight}

class RandomVFlip(object):
    
    def __init__(self, prob=0.5 ): 
        self.prob = prob

    def __call__(self, sample):            
        image, label, weight = sample['image'], sample['label'], sample['weight']

        if random.random() < self.prob:    
            image, label, weight = [ cv2.flip(x,0) for x in [image, label, weight] ]
            image  = cunsqueeze(image)
            label  = cunsqueeze(label)
            weight  = cunsqueeze(weight)
        return {'image': image, 'label': label, 'weight': weight}

class GeometricDistort(object):
    '''
    Geometric transformation 
    '''

    def __init__(self, angle=360, translation=0.2, warp=0.0, prob=0.5 ): 
        self.angle = angle
        self.translation = translation
        self.warp = warp
        self.prob = prob

    def __call__(self, sample):
        
        image, label, weight = sample['image'], sample['label'], sample['weight']

        if random.random() < self.prob:

            bfliplr = random.random() < 0.5
            bflipud = random.random() < 0.5

            # apply tranform
            rotation_mat, translation_mat, warp_mat  = geometric_transform( 
                image.shape, 
                self.angle,
                self.translation,
                self.warp,
                )

            height, width = image.shape[:2]

            #borderMode=cv2.BORDER_REFLECT_101
            #flags=cv2.INTER_LINEAR

            image_t = image
            image_t = cv2.warpAffine(image_t, rotation_mat, (width, height), flags=cv2.INTER_LINEAR )
            image_t = cv2.warpAffine(image_t, translation_mat, (width, height), flags=cv2.INTER_LINEAR )
            #image_t = cv2.warpAffine(image_t, warp_mat, (width, height) )
            if bfliplr: image_t = cv2.flip(image_t,0)
            if bflipud: image_t = cv2.flip(image_t,1)
        
            label_t  = label
            label_t  = cv2.warpAffine(label_t, rotation_mat, (width, height), flags=cv2.INTER_NEAREST )
            label_t  = cv2.warpAffine(label_t, translation_mat, (width, height), flags=cv2.INTER_NEAREST )
            #label_t = cv2.warpAffine(label_t, warp_mat, (width, height))
            if bfliplr: label_t = cv2.flip(label_t,0)
            if bflipud: label_t = cv2.flip(label_t,1)

            weight_t = weight
            weight_t = cv2.warpAffine(weight_t, rotation_mat, (width, height), flags=cv2.INTER_LINEAR )
            weight_t = cv2.warpAffine(weight_t, translation_mat, (width, height), flags=cv2.INTER_LINEAR ) 
            #weight_t = cv2.warpAffine(weight_t, warp_mat, (width, height))
            if bfliplr: weight_t = cv2.flip(weight_t,0)
            if bflipud: weight_t = cv2.flip(weight_t,1)

            area = label[:,:,1].sum()
            area_t = label_t[:,:,1].sum()
            if area_t/area > 0.75 and area_t > 50: #5x5x2 
                image  = cunsqueeze(image_t)
                label  = cunsqueeze(label_t)
                weight = cunsqueeze(weight_t)

        return {'image': image, 'label': label, 'weight': weight}



class ElasticDistort(object):
    '''
    Elastic transformation 
    '''

    def __init__(self, size_grid=50, deform=15, prob=0.5): 
        self.size_grid = size_grid
        self.deform = deform
        self.prob = prob

    def __call__(self, sample):
        
        image, label, weight = sample['image'], sample['label'], sample['weight']
        
        if random.random() < self.prob:

            # get transform
            mapx, mapy = elastic_transform(image.shape, self.size_grid, self.deform )

            # apply tranform
            image  = cv2.remap(image,  mapx, mapy, cv2.INTER_CUBIC)
            label  = cv2.remap(label,  mapx, mapy, cv2.INTER_NEAREST)
            weight = cv2.remap(weight, mapx, mapy, cv2.INTER_CUBIC)
        
        return {'image': image, 'label': label, 'weight': weight }

class ElasticTorchDistort(object):
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
            label_t  = grid_sample(torch.unsqueeze(label.float(),dim=0) , grid).round().data[0,...]
            weight_t = grid_sample(torch.unsqueeze(weight,dim=0), grid).data[0,...]

            if label_t[1,...].sum() > 0:
                image = image_t;
                label = label_t;
                weight = weight_t;

                    
        return {'image': image, 'label': label, 'weight': weight}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, weight = sample['image'], sample['label'], sample['weight'] 
        label = (label>0).astype(np.uint8)
        
        # label_t = np.zeros( (label.shape[0],label.shape[1],3) )
        # label_t[:,:,0] = label[:,:,0]==0
        # label_t[:,:,1] = label[:,:,0]==1
        # label_t[:,:,2] = label[:,:,0]==2
        # label = label_t

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W        
        image  = image.transpose((2, 0, 1)).astype(np.float)
        label  = label.transpose((2, 0, 1)).astype(np.float)
        weight = weight.transpose((2, 0, 1)).astype(np.float)

        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label).float(),
                'weight': torch.from_numpy(weight).float(),
               }



#//////////////////////////////////////////////////////////////////////////
# Color Tranformers
# https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/albu/src/transforms.py
# https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/image/image.py


## filter ====================================================================================

def do_unsharp(image, size=9, strength=0.25, alpha=5 ):
    image = image.astype(np.float32)
    size  = 1+2*(int(size)//2)
    strength = strength*255
    blur  = cv2.GaussianBlur(image, (size,size), strength)
    image = alpha*image + (1-alpha)*blur
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

#noise
def do_gaussian_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H,W  = gray.shape

    noise = np.random.normal(0,sigma,(H,W))
    noisy = gray + noise

    noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)
    lab   = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image

def do_speckle_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H,W  = gray.shape

    noise = sigma*np.random.randn(H,W)
    noisy = gray + gray * noise

    noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)
    lab   = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image

def do_inv_speckle_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H,W  = gray.shape

    noise = sigma*np.random.randn(H,W)
    noisy = gray + (1-gray) * noise

    noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)
    lab   = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image

# elif noise_type == "salt_and_pepper":
#     salt_pepper_ratio = 0.5
#     noisy    = np.copy(gray)
#
#     num_salt = np.ceil(limit * H*W * salt_pepper_ratio)
#     coords = [np.random.randint(0, i - 1, int(num_salt)) for i in gray.shape]
#     noisy[coords] = 1
#
#     num_pepper = np.ceil(amount* H*W * (1. - salt_pepper_ratio))
#     coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in gray.shape]
#     noisy[coords] = 0
#
# elif noise_type == "poisson":
#     num_values = len(np.unique(gray))
#     num_values = 2 ** np.ceil(np.log2(num_values))
#     noisy      = np.random.poisson(gray * num_values) / float(num_values)


# # COLOR SEGMENTATION
class RandomGaussianBlur(object):
    
    def __init__(self, prob=0.5):
        self.prob = prob     

    def __call__(self, image):        

        if random.random() < self.prob:

            row,col,ch = image.shape
            # mean = 0
            # var = 0.1
            # sigma = var**0.5
            # gauss = np.array([random.gauss(mean,sigma) for i in range(row*col*ch)])
            # gauss = gauss.reshape(row,col,ch)
            # image = image + 10*(gauss)
            # image = np.uint8(np.clip(image,0,255)) 

            image = do_gaussian_noise(image, 0.01**0.5  )
            wnd = random.randint(1,3) * 2 + 1
            image = cv2.GaussianBlur(image, (wnd, wnd), 0);             

        return image

class RandomFilter:
    """
    blur sharpen, etc
    """
    def __init__(self, limit=.5, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = self.limit * random.uniform(0, 1)
            kernel = np.ones((3, 3), np.float32)/9 * 0.2

            colored = img[..., :3]
            colored = alpha * cv2.filter2D(colored, -1, kernel) + (1-alpha) * colored
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(colored, dtype, maxval)

        return img


# brightness, contrast, saturation-------------
# from mxnet code, see: https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/image/image.py

## illumination ====================================================================================

class RandomBrightness:
    def __init__(self, limit=0.1, prob=0.5):
        self.limit = limit
        self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit*random.uniform(-1, 1)
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(alpha * img[...,:3].astype(np.float32), dtype, maxval)
        return img

class RandomBrightnessShift:
    def __init__(self, limit=0.125, prob=0.5):
        self.limit = limit
        self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit*random.uniform(-1, 1)
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(alpha * 255 + img[...,:3].astype(np.float32), dtype, maxval)
        return img

class RandomContrast:
    def __init__(self, limit=0.1, prob=.5):
        self.limit = limit
        self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit*random.uniform(-1, 1)
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
            gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[:, :, :3] = clip(alpha * img[:, :, :3].astype(np.float32) + gray, dtype, maxval)
        return img


#https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
class RandomGamma:
    def __init__(self, limit=0.5, prob=.5):
        self.limit = limit
        self.prob = prob
    def __call__(self, image):        

        if random.random() < self.prob:     
            gamma = 1.0 + self.limit*random.uniform(-1, 1)      
            table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
		        for i in np.arange(0, 256)]).astype("uint8")
            image = cv2.LUT(image, table) # apply gamma correction using the lookup table
        return image


class CLAHE(object):
    
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, im):
        img_yuv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return img_output

## color ====================================================================================


class RandomSaturation:
    def __init__(self, limit=0.3, prob=0.5):
        self.limit = limit
        self.prob = prob
    def __call__(self, img):
        # dont work :(
        if random.random() < self.prob:
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            alpha = 1.0 + random.uniform(-self.limit, self.limit)
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
    Color distortion and normalization
    '''

    def __init__( self ):
        self.randomBrightness = RandomBrightness()
        self.randomHueSaturationValue = RandomHueSaturationValue(  )
        self.randomContrast = RandomContrast()
        self.clahe = CLAHE()
        self.randomGaussianBlur = RandomGaussianBlur()
        self.negative = Negative()
        self.gray = Grayscale()
        self.colorchange = ColorPermutation()
        self.randomHueSaturationShift = RandomHueSaturationShift()
        self.randomGamma = RandomGamma()
        self.randomBrightnessShift = RandomBrightnessShift()

    def __call__(self, sample):
        
        image, label, weight = sample['image'], sample['label'], sample['weight']
        
        #RandomBrightness, RandomHueSaturationValue
        #RandomContrast, RandomFilter

        # apply tranform
        image = self.randomBrightness(image)
        image = self.randomHueSaturationValue(image)
        image = self.randomHueSaturationShift(image)
        image = self.randomContrast(image)  
        image = self.randomGamma(image) 
        ####image = self.randomBrightnessShift(image)      
        
        image = self.colorchange(image)
        image = self.negative(image)
        image = self.gray(image)
        image = self.randomGaussianBlur(image)        
        #image = self.clahe(image)
        
        return {'image': image, 'label': label, 'weight': weight}



def tnormalize(tensor, mean, std):
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



class Normalize(object):
    '''
    Color distortion and normalization
    '''

    def __init__( self ):
        pass

    def __call__(self, sample):
        
        image, label, weight = sample['image'], sample['label'], sample['weight']

        new_image = []
        for im in image:
            im = im.sub_( im.min() )
            im = im.div_( im.max() )
            new_image.append( im )        
        image = torch.stack(new_image, 0)  

        #image = image.mean( dim=0 ).unsqueeze(dim=0)
       
        return {'image': image, 'label': label, 'weight': weight}