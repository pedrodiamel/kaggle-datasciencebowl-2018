

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

def to_unsharp(image, size=9, strength=0.25, alpha=5 ):
    
    image = image.astype(np.float32)
    size  = 1+2*(int(size)//2)
    strength = strength*255
    blur  = cv2.GaussianBlur(image, (size,size), strength)
    image = alpha*image + (1-alpha)*blur
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image


def to_gaussian_noise(image, sigma=0.5):
    

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    gray, a, b = cv2.split(lab)    
    gray = gray.astype(np.float32)/255
    
    H,W  = gray.shape
    noise = np.array([random.gauss(0,sigma) for i in range(H*W)])
    noise = noise.reshape(H,W)
    noisy = gray + noise
    noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)

    lab   = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return image


def to_speckle_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H,W  = gray.shape

    noise = np.array([random.random() for i in range(H*W)])
    noise = noise.reshape(H,W)
    noisy = gray + gray * noise

    noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)
    lab   = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image


def do_inv_speckle_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H,W  = gray.shape

    noise = sigma*random.randn(H,W)
    noise = np.array([random.random() for i in range(H*W)])
    noise = noise.reshape(H,W)
    noisy = gray + (1-gray) * noise

    noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)
    lab   = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image



def imagecrop( image, cropsize, top, left ):
    
    #if mult channel
    bchannel = False
    if len(image.shape) != 3:
        image = image[:,:,np.newaxis ]
        bchannel = True
    
    h, w, c = image.shape
    new_h, new_w = cropsize
    imagecrop = image[top:top + new_h, left:left + new_w, : ]
    
    if bchannel: imagecrop = imagecrop[:,:,0]
    return imagecrop



def to_one_hot( x, nc ):
    y = np.zeros((nc)); y[x] = 1.0
    return y

def summary(image):
    print(image.shape, image.min(), image.max())

def norm_fro(a,b): 
    return np.sum( (a-b)**2.0 );

def complex2vector(c):
    '''complex to vector'''    
    return np.concatenate( ( c.real, c.imag ) , axis=1 )

def ffftshift2(h):    
    H = np.fft.fft2(h)
    H = np.abs( np.fft.fftshift( H ) )
    return H


def image_to_array(image, channels=None):
    """
    Returns an image as a np.array

    Arguments:
    image -- a PIL.Image or numpy.ndarray

    Keyword Arguments:
    channels -- channels of new image (stays unchanged if not specified)
    """

    if channels not in [None, 1, 3, 4]:
        raise ValueError('unsupported number of channels: %s' % channels)

    if isinstance(image, PIL.Image.Image):
        # Convert image mode (channels)
        if channels is None:
            image_mode = image.mode
            if image_mode not in ['L', 'RGB', 'RGBA']:
                raise ValueError('unknown image mode "%s"' % image_mode)
        elif channels == 1:
            # 8-bit pixels, black and white
            image_mode = 'L'
        elif channels == 3:
            # 3x8-bit pixels, true color
            image_mode = 'RGB'
        elif channels == 4:
            # 4x8-bit pixels, true color with alpha
            image_mode = 'RGBA'
        if image.mode != image_mode:
            image = image.convert(image_mode)
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.reshape(image.shape[:2])
        if channels is None:
            if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] in [3, 4])):
                raise ValueError('invalid image shape: %s' % (image.shape,))
        elif channels == 1:
            if image.ndim != 2:
                if image.ndim == 3 and image.shape[2] in [3, 4]:
                    # color to grayscale. throw away alpha
                    image = np.dot(image[:, :, :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                else:
                    raise ValueError('invalid image shape: %s' % (image.shape,))
        elif channels == 3:
            if image.ndim == 2:
                # grayscale to color
                image = np.repeat(image, 3).reshape(image.shape + (3,))
            elif image.shape[2] == 4:
                # throw away alpha
                image = image[:, :, :3]
            elif image.shape[2] != 3:
                raise ValueError('invalid image shape: %s' % (image.shape,))
        elif channels == 4:
            if image.ndim == 2:
                # grayscale to color
                image = np.repeat(image, 4).reshape(image.shape + (4,))
                image[:, :, 3] = 255
            elif image.shape[2] == 3:
                # add alpha
                image = np.append(image, np.zeros(image.shape[:2] + (1,), dtype='uint8'), axis=2)
                image[:, :, 3] = 255
            elif image.shape[2] != 4:
                raise ValueError('invalid image shape: %s' % (image.shape,))
    else:
        raise ValueError('resize_image() expected a PIL.Image.Image or a numpy.ndarray')

    return image



def resize_image(image, height, width,
                 channels=None,
                 resize_mode=None,
                 ):
    """
    Resizes an image and returns it as a np.array

    Arguments:
    image -- a PIL.Image or numpy.ndarray
    height -- height of new image
    width -- width of new image

    Keyword Arguments:
    channels -- channels of new image (stays unchanged if not specified)
    resize_mode -- can be crop, squash, fill or half_crop
    """

    if resize_mode is None:
        resize_mode = 'squash'
    if resize_mode not in ['crop', 'squash', 'fill', 'half_crop']:
        raise ValueError('resize_mode "%s" not supported' % resize_mode)

    # convert to array
    image = image_to_array(image, channels)

    # No need to resize
    if image.shape[0] == height and image.shape[1] == width:
        return image

    # Resize
    interp = 'bilinear'

    width_ratio = float(image.shape[1]) / width
    height_ratio = float(image.shape[0]) / height
    if resize_mode == 'squash' or width_ratio == height_ratio:
        return scipy.misc.imresize(image, (height, width), interp=interp)
    elif resize_mode == 'crop':
        # resize to smallest of ratios (relatively larger image), keeping aspect ratio
        if width_ratio > height_ratio:
            resize_height = height
            resize_width = int(round(image.shape[1] / height_ratio))
        else:
            resize_width = width
            resize_height = int(round(image.shape[0] / width_ratio))
        image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)

        # chop off ends of dimension that is still too long
        if width_ratio > height_ratio:
            start = int(round((resize_width - width) / 2.0))
            return image[:, start:start + width]
        else:
            start = int(round((resize_height - height) / 2.0))
            return image[start:start + height, :]
    else:
        if resize_mode == 'fill':
            # resize to biggest of ratios (relatively smaller image), keeping aspect ratio
            if width_ratio > height_ratio:
                resize_width = width
                resize_height = int(round(image.shape[0] / width_ratio))
                if (height - resize_height) % 2 == 1:
                    resize_height += 1
            else:
                resize_height = height
                resize_width = int(round(image.shape[1] / height_ratio))
                if (width - resize_width) % 2 == 1:
                    resize_width += 1
            image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)
        elif resize_mode == 'half_crop':
            # resize to average ratio keeping aspect ratio
            new_ratio = (width_ratio + height_ratio) / 2.0
            resize_width = int(round(image.shape[1] / new_ratio))
            resize_height = int(round(image.shape[0] / new_ratio))
            if width_ratio > height_ratio and (height - resize_height) % 2 == 1:
                resize_height += 1
            elif width_ratio < height_ratio and (width - resize_width) % 2 == 1:
                resize_width += 1
            image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)
            # chop off ends of dimension that is still too long
            if width_ratio > height_ratio:
                start = int(round((resize_width - width) / 2.0))
                image = image[:, start:start + width]
            else:
                start = int(round((resize_height - height) / 2.0))
                image = image[start:start + height, :]
        else:
            raise Exception('unrecognized resize_mode "%s"' % resize_mode)

        # fill ends of dimension that is too short with random noise
        if width_ratio > height_ratio:
            padding = (height - resize_height) / 2
            noise_size = (padding, width)
            if channels > 1:
                noise_size += (channels,)
            noise = np.random.randint(0, 255, noise_size).astype('uint8')
            image = np.concatenate((noise, image, noise), axis=0)
        else:
            padding = (width - resize_width) / 2
            noise_size = (height, padding)
            if channels > 1:
                noise_size += (channels,)
            noise = np.random.randint(0, 255, noise_size).astype('uint8')
            image = np.concatenate((noise, image, noise), axis=1)

        return image
