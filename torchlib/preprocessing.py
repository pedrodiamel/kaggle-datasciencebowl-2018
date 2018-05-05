
import os
import numpy as np
from skimage import color
import scipy.misc
from scipy import ndimage

import cv2
import random
import csv
import h5py

from skimage import io, transform, morphology, filters
from scipy import ndimage
import skimage.morphology as morph
import skfmm

from deep.datasets import weightmaps 

def tolabel( x ):
    return (np.max(x,axis=0)>0) 

def summary(data):
    print(np.min(data), np.max(data), data.shape)

def onehot2label( labels ):
    #classe 0 back
    m,n,c = labels.shape
    mlabel = np.zeros( (m,n) )
    for i in range(c):
        mlabel += labels[:,:,i]*(i+1)
    return mlabel

def get_contour(img):    
    img = img.astype(np.uint8)
    edge = np.zeros_like(img)
    _,cnt,_ = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE )
    cv2.drawContours( edge, cnt, -1, 1 , 1)
    edge = (edge>0).astype(np.uint8)    
    return edge

def get_center(img):
    cent = np.zeros_like(img).astype(np.uint8)
    y, x = ndimage.measurements.center_of_mass(img)
    cv2.circle(cent, (int(x), int(y)), 1, 1, -1)
    cent = (cent>0).astype(np.uint8) 
    return cent

def get_distance(x):
    return skfmm.distance((x).astype('float32') - 0.5) 

def get_touchs( edges ):       
    A = np.array([ morph.binary_dilation( c, morph.square(3) )  for c in edges ]) 
    A = np.sum(A,axis=0)>1  
    I = morph.remove_small_objects( A, 3 )
    I = morph.skeletonize(I)
    I = morph.binary_dilation( I, morph.square(3) )    
    return I

def imagecrop( image, cropsize, top, left ):
    
    #if mult channel
    bchannel = False
    if len(image.shape) != 3:
        image = image[:,:,np.newaxis ]
        bchannel = True
    
    h, w, c = image.shape
    new_h, new_w = cropsize
    imagecrop = image[top:top + new_h, left:left + new_w, : ]
    
    if bchannel:
        imagecrop = imagecrop[:,:,0]
    
    return imagecrop

def randomcrop(image, label, cropsize):
    
    h,w = image.shape[:2]
    new_h, new_w = cropsize
    
    print(h,w, new_h, new_w)

    top  = np.random.randint( h - new_h ) 
    left = np.random.randint( w - new_w ) 

    image = imagecrop( image, cropsize, top, left)
    label = imagecrop( label, cropsize, top, left)
    
    return image, label


def size_transform(imagein, size=512, mode=None):
        
    image = np.array(imagein.copy())
    height, width = image.shape[:2];
    asp = float(height)/width;
    w = size; 
    h = int(w*asp);       
    
    if len(image.shape) == 2:
        image_x = scipy.misc.imresize(image, (h,w), interp='bilinear', mode=mode); 
        image = np.zeros((w,w));
    else: 
        image_x = scipy.misc.imresize(image, (h,w), interp='bilinear', mode=mode); 
        image = np.zeros((w,w,3));

    ini = int(round((w-h)/2.0));
    image[ini:h+ini,...] = image_x;

    return image


def create_groundtruth(masks):
    
    edges = np.array([ morph.binary_dilation(get_contour(x)) for x in masks ])   
    #edges = np.array([ (get_contour(x)) for x in masks ]) 
    
    bmask = tolabel(masks)  
    bedge = tolabel(edges)   
    
    btouch   = get_touchs( edges )
    bcontour = tolabel(edges)

    centers  = np.array([ morph.binary_dilation(get_center(x)) for x in masks ]) 
    bcenters = tolabel(centers)   
    
    bmask = bmask*(~btouch)
    #bmask = bmask*(~bcontour)
    
    # label = np.zeros((bmask.shape[0],bmask.shape[1],3))
    # label[:,:,0] = bmask
    # label[:,:,1] = bcontour #btouch
    # label[:,:,2] = bcenters
    
    return bmask, bcontour, btouch, bcenters


def delete_black_layer( masks ):    
    newmasks = []
    for mask in masks:
        if mask.sum() != 0:
            newmasks.append(mask)             
    return newmasks

def preprocessing(image, label, imsize=250, bcrop=False):
    

    if bcrop:        
        barea = False
        while barea==False:
            image_t, label_t = randomcrop(image, label, (imsize,imsize) )
            masks = (label_t.transpose((2,0,1))>0).astype(np.uint32)
            masks = np.array([ ndimage.morphology.binary_fill_holes(x) for x in masks ])
            masks = delete_black_layer(masks)
            barea = np.sum(masks) > 0
        image = image_t
        label = label_t
    else:        
        masks = (label.transpose((2,0,1))>0).astype(np.uint32)
        masks = np.array([ ndimage.morphology.binary_fill_holes(x) for x in masks ])
    
    # preprocessing
    # image =  color.rgb2gray(image)
    bmask, bcontour, btouch, bcenters = create_groundtruth(masks)
    weight = weightmaps.getunetweightmap( bmask + 2*btouch  , masks, w0=10, sigma=5, )

    # resize
    #image_t   = size_transform(image,  size=imsize, mode=None)
    #label_t   = size_transform(label,  size=imsize, mode='RGB')
    #weight_t  = size_transform(weight, size=imsize, mode='F')

    return image, bmask, bcontour, btouch, bcenters, weight
