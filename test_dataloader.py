
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import csv
from skimage import color
import scipy.misc

from deep.datasets import imageutl as imutl
from deep.datasets import utility as utl
from deep import visualization as view
from deep.datasets import dsxbdata 
from deep.datasets import dsxbtransform as dsxbtrans



pathdataset     = '../db'
namedataset     = 'databoewlex'
sub_folder      =  'train'
folders_images  = 'images'
folders_labels  = 'labels'
folders_weights = 'weights'

base_folder = os.path.join(pathdataset, namedataset) 


data = dsxbdata.DSXBDataset(
        base_folder, 
        sub_folder, 
        folders_contours='touchs',
        transform=transforms.Compose([
            #dsxbtrans.ElasticDistort(size_grid=50, deform=15),                       
            dsxbtrans.RandomCrop( cropsize=(50,50) ),
            #dsxbtrans.ColorDistort(),
            #dsxbtrans.ShiftScale(prob=1.0, limit=20), 
            #dsxbtrans.GeometricDistort(angle=360, translation=0.05, warp=0.01),            
            #dsxbtrans.RandomFlip(prob=0.75),
            dsxbtrans.UnetResize(imsize=128),                     
            dsxbtrans.ToTensor(),
            #dsxbtrans.ElasticTorchDistort(size_grid=10, deform=0.05),
            dsxbtrans.Normalize(),  
            ])
        )

dataloader = DataLoader(data, batch_size=3, shuffle=True, num_workers=1 )

label_batched = []
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['label'].size(),
          sample_batched['weight'].size()    
         )
    
    image_a = sample_batched['image'][0,:,...]
    image_b = sample_batched['image'][1,:,...]
    image_c = sample_batched['image'][2,:,...]

    image = sample_batched['image'][0,0,...]
    label = sample_batched['label'][0,2,...]
    weight = sample_batched['weight'][0,0,...]
    
    print(torch.min(image), torch.max(image), image.shape )
    print(torch.min(label), torch.max(label), image.shape )
    print(torch.min(weight), torch.max(weight), image.shape )

    print(image_a.shape)
    print( np.unique(label) )
        
    plt.figure( figsize=(15,15) )
    plt.subplot(131)
    plt.imshow( image_a.permute(1,2,0).squeeze()) #, cmap='gray' 
    plt.axis('off')
    plt.ioff()

    plt.subplot(132)
    plt.imshow( image_b.permute(1,2,0).squeeze() ) 
    #plt.imshow( label ) #cmap='gray'
    plt.axis('off')
    plt.ioff()

    plt.subplot(133)
    plt.imshow( image_c.permute(1,2,0).squeeze()  ) 
    #plt.imshow( weight )
    plt.axis('off')

    plt.ioff()       
    plt.show()        

    # observe 4th batch and stop.
    if i_batch == 4: 
        break        





