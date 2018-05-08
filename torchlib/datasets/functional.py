

import os
import sys
import numpy as np
import PIL.Image

import math
import cv2
import random
import csv
import h5py


import skimage.morphology as morph
from skimage import io, transform, morphology, filters
import skimage.color as skcolor
import skimage.util as skutl

from scipy import ndimage
import scipy.misc
import skfmm







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

def unsharp(image, size=9, strength=0.25, alpha=5 ):
    
    image = image.astype(np.float32)
    size  = 1+2*(int(size)//2)
    strength = strength*255
    blur  = cv2.GaussianBlur(image, (size,size), strength)
    image = alpha*image + (1-alpha)*blur
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image


def gaussian_noise(image, sigma=0.5):
    
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


def speckle_noise(image, sigma=0.5):
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


def inv_speckle_noise(image, sigma=0.5):
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



# ELASTIC TRANSFORMATION
def elastic_transform(shape, size_grid, deform):
    """Get elastic tranform 
    Args:
        @shape: image shape
        @size_grid: size of the grid (example (10,10) )
        @deform: deform coeficient
    """
        
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
    """Get elastic tranform for tensor
    Args:
        @shape: image shape
        @size_grid: size of the grid (example (10,10) )
        @deform: deform coeficient
    """   
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
    """Transform the image for data augmentation
    Args:
        @degree: Max rotation angle, in degrees. Direction of rotation is random.
        @translation: Max translation amount in both x and y directions,
            expressed as fraction of total image width/height
        @warp: Max warp amount for each of the 3 reference points,
            expressed as fraction of total image width/height

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



def square_resize(image, newsize, interpolate_mode, padding_mode):
    
    w, h, channels = image.shape; 
    if w == h: return image  

    if w>h:        
        padxL = int(np.floor( (w-h) / 2.0));
        padxR = int(np.ceil( (w-h) / 2.0)) ;
        padyT, padyB = 0,0 
    else:
        padxL, padxR = 0,0;
        padyT = int(np.floor( (h-w) / 2.0));
        padyB = int(np.ceil( (h-w) / 2.0));

    image = cv2.copyMakeBorder(image, padxL, padxR, padyT, padyB, borderType=padding_mode)
    image = cv2.resize(image, (newsize, newsize) , interpolation = interpolate_mode)    

    image = cunsqueeze(image)
    return image



# UNET RESIZE
def resize_unet_transform(imagein, size, interpolate_mode, padding_mode): 
    
    height, width, ch = imagein.shape
    image = np.array(imagein.copy())
    
    if height < width:
        asp = float(height)/width
        w = size
        h = int(w*asp)
    else:
        asp = float(width)/height 
        h = size
        w = int(h*asp) 
        
    #resize mantaining aspect ratio
    image = cv2.resize(image, (w,h) , interpolation = interpolate_mode)
    image = cunsqueeze(image)

    #unet required input size
    downsampleFactor = 16
    d4a_size   = 0
    padInput   = (((d4a_size *2 +2 +2)*2 +2 +2)*2 +2 +2)*2 +2 +2
    padOutput  = ((((d4a_size -2 -2)*2-2 -2)*2-2 -2)*2-2 -2)*2-2 -2
    d4a_size   = math.ceil( (size - padOutput)/downsampleFactor)
    input_size = downsampleFactor*d4a_size + padInput

    image = square_resize(image, input_size, interpolate_mode, padding_mode)
    
    return image







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
