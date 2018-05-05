import os
import torch
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import scipy.misc
from skimage import color
from skimage import io, transform
import cv2

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as nnfun

from deep.datasets import imageutl as imutl
from deep.datasets import utility as utl
from deep.datasets import weightmaps 
from deep import netmodels as nnmodels
from deep import visualization as view
from deep import netutility as nutl
from deep import neuralnet as deepnet
from deep import postprocessing as posp
from deep import processing as proc

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# configuration
PATHDATASET = '../db'
NAMEDATASET = 'databoewl'
METADATA = 'stage1_train_labels.csv'
PATHMODEL = 'netruns/experiment_unet_dballex_dim388_lr0001_ep1000_adam_b12_wk12_pll_wmce_contours_c0001'
NAMEMODEL = 'chk000205.pth.tar'
RLNAME = 'test_masks.csv'
NUMITER = 5
PATHNAMEDATASET = os.path.join(PATHDATASET, NAMEDATASET);
PATHNAMEDMETADATA = os.path.join(PATHDATASET, NAMEDATASET, METADATA)
PATHNAMEMODEL = os.path.join(PATHMODEL, NAMEMODEL)


def imageshowlist(image_in, image_out ):    
    plt.figure( figsize=(16,16))
    plt.subplot(121)
    plt.imshow( image_in )
    plt.title('Image input')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow( image_out )
    plt.title('Image output')
    plt.axis('off')
    plt.show()

def imageshow( image, title='image'):
    plt.figure( figsize=(8,8))
    plt.imshow( image )
    plt.axis('off')
    plt.show()


base_folder = PATHNAMEDATASET
sub_folder =  imutl.testfile
folders_image='images'

dataloader = imutl.dsxbImageProvide.create(
    base_folder, 
    sub_folder, 
    folders_image, 
    )

print(len(dataloader))
print(':)!!!')


segment = proc.Net( ntiles=3 )
segment.loadmodel( PATHNAMEMODEL )

i= 20 #49 #24 #32 #20
image = dataloader[ i ]
nutl.summary(image)

score = segment(image)
predition = np.argmax(score, axis=2).astype('uint8') 

score_prob = nutl.sigmoid(score)
labels_est = posp.mpostprocess(score)
labels_mask_est = np.transpose( labels_est, (1,2,0) )

labels_mask_est = labels_mask_est[:,:, np.random.permutation(labels_mask_est.shape[2]) ]
imagecell_est = view.makeimagecell(image, labels_mask_est, alphaback=0.3, alphaedge=0.9)
imageshowlist(image, imagecell_est )
plt.show()