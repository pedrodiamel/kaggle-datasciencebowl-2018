

import os
import sys
import numpy as np
import PIL.Image

import math
import cv2
import random

from scipy.interpolate import griddata

from scipy import ndimage
import scipy.misc

import torch
from torch.autograd import Variable

import itertools

from .grid.grid_sample import grid_sample
from .grid.tps_grid_gen import TPSGridGen


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

def scale(image, factor, mode, padding_mode ): 

    h,w = image.shape[:2]
    image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=mode ) 
    image = cunsqueeze(image)
    hn, wn = image.shape[:2]
    borderX = float( abs(wn-w) )/2.0
    borderY = float( abs(hn-h) )/2.0
    padxL = int(np.floor( borderY ))
    padxR = int(np.ceil(  borderY ))  
    padyT = int(np.floor( borderX ))
    padyB = int(np.ceil(  borderX ))

    if sxy < 1:  image = cv2.copyMakeBorder(image, padxL, padxR, padyT, padyB, borderType=padding_mode)
    else: image = image[ padyT:padyT+h, padxL:padxL+w, : ]

    image = cunsqueeze(image)    
    return image

def hflip( x ): 
    return cv2.flip(x,1)

def vflip( x ):
    return cv2.flip(x,0)

def rotate90( x ): 
    return cv2.flip(x.transpose(1,0,2),1)

def rotate180( x ): 
    return cv2.flip(x,-1)

def rotate270( x ):
    return cv2.flip(x.transpose(1,0,2),0)

def is_box_inside(img, box ):
    return box[0] < 0 or box[1] < 0 or box[2]+box[0] >= img.shape[1] or box[3]+box[1] >= img.shape[0]

def pad_img_to_fit_bbox(img, box, padding_mode):
    
    x1,y1,x2,y2 = box
    x2 = x1+x2; y2 = y1+y2

    padxL = np.abs(np.minimum(0, y1))
    padxR = np.maximum(y2 - img.shape[0], 0)
    padyT = (np.abs(np.minimum(0, x1)))
    padyB = np.maximum(x2 - img.shape[1], 0)
    img = cv2.copyMakeBorder(img, padxL, padxR, padyT, padyB, borderType=padding_mode )
    
    # img = np.pad(
    #     img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
    #           (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), 
    #           (0,0)), mode="constant")  

    img = cunsqueeze(img)
    y2 += np.abs(np.minimum(0, y1))
    y1 += np.abs(np.minimum(0, y1))
    x2 += np.abs(np.minimum(0, x1))
    x1 += np.abs(np.minimum(0, x1))

    return img, [ x1, y1, x2-x1, y2-y1 ]

def imcrop( image, box, padding_mode ):
    """ Image crop
    Args
        @image
        @box: [x,y,w,h]
    """    
    h, w, c = image.shape
    if is_box_inside(image, box):
        image, box = pad_img_to_fit_bbox(image, box, padding_mode)    
    x, y, new_w, new_h = box
    imagecrop = image[y:y + new_h, x:x + new_w, : ]   
    imagecrop = cunsqueeze(imagecrop)

    return imagecrop

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

def get_elastic_transform(shape, size_grid, deform):
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

def get_tensor_elastic_transform( shape, size_grid, deform):
    """Get elastic tranform for tensor
    Args:
        @shape: image shape
        @size_grid: size of the grid (example (10,10) )
        @deform: deform coeficient
    """   
    target_height, target_width = shape[:2]

    target_control_points = torch.Tensor( list(itertools.product(
        torch.arange(-1.0, 1.00001, 2.0 / (size_grid-1)),
        torch.arange(-1.0, 1.00001, 2.0 / (size_grid-1)),
        )))

    source_control_points = target_control_points + torch.Tensor(target_control_points.size()).uniform_(-deform, deform)
    tps = TPSGridGen(target_height, target_width, target_control_points)
    source_coordinate = tps(Variable(torch.unsqueeze(source_control_points, 0)))
    grid = source_coordinate.view(1, target_height, target_width, 2)
    
    return grid

def get_geometric_random_transform( imsize, degree, translation, warp ):
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

def applay_geometrical_transform( image, mat_r, mat_t, mat_w, interpolate_mode, padding_mode  ):
    h,w = image.shape[:2] 
    image = cv2.warpAffine(image, mat_r, (w,h), flags=interpolate_mode, borderMode=padding_mode )
    image = cv2.warpAffine(image, mat_t, (w,h), flags=interpolate_mode, borderMode=padding_mode )
    image = cv2.warpAffine(image, mat_w, (w,h), flags=interpolate_mode, borderMode=padding_mode )
    image = cunsqueeze(image)
    return image

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



def draw_grid(imgrid, grid_size=50, color=(255,0,0), thickness=1):
    
    m,n = imgrid.shape[:2]
    # Draw grid lines
    for i in range(0, n-1, grid_size):
        cv2.line(imgrid, (i+grid_size, 0), (i+grid_size, m), color=color, thickness=thickness)
    for j in range(0, m-1, grid_size):
        cv2.line(imgrid, (0, j+grid_size), (n, j+grid_size), color=color, thickness=thickness)
    return imgrid


def resize_unet_transform(image, size, interpolate_mode, padding_mode): 
    
    height, width, ch = image.shape
    
    #unet required input size
    downsampleFactor = 16
    d4a_size   = 0
    padInput   = (((d4a_size *2 +2 +2)*2 +2 +2)*2 +2 +2)*2 +2 +2
    padOutput  = ((((d4a_size -2 -2)*2-2 -2)*2-2 -2)*2-2 -2)*2-2 -2    
    d4a_size   = math.ceil( (size - padOutput)/downsampleFactor)
    input_size  = downsampleFactor*d4a_size + padInput
    output_size = downsampleFactor*d4a_size + padOutput;
    
    if height < width:
        asp = float(height)/width
        w = output_size
        h = int(w*asp)
    else:
        asp = float(width)/height 
        h = output_size
        w = int(h*asp) 
        
    #resize mantaining aspect ratio
    image = cv2.resize(image, (w,h), interpolation = interpolate_mode)
    image = cunsqueeze(image)

    borderX = float(input_size-w)/2.0
    borderY = float(input_size-h)/2.0

    padxL = int(np.floor( borderY ))
    padxR = int(np.ceil(  borderY )) 
    padyT = int(np.floor( borderX ))
    padyB = int(np.ceil(  borderX )) 

    #image = square_resize(image, input_size, interpolate_mode, padding_mode)
    image = cv2.copyMakeBorder(image, padxL, padxR, padyT, padyB, borderType=padding_mode)
    image = cv2.resize(image, (input_size, input_size) , interpolation = interpolate_mode)    
    image = cunsqueeze(image)

    return image


def ffftshift2(h):    
    H = np.fft.fft2(h)
    H = np.abs( np.fft.fftshift( H ) )
    return H

def norm_fro(a,b): 
    return np.sum( (a-b)**2.0 );

def complex2vector(c):
    '''complex to vector'''    
    return np.concatenate( ( c.real, c.imag ) , axis=1 )


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