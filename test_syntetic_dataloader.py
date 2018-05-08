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
import cv2

from torchlib.datasets.syntheticdata import SynteticCircleDataset
from torchlib.datasets import transforms as mtrans

from torchlib.datasets import imageutl as imutl
from torchlib.datasets import utility as utl
from torchlib import visualization as view


data = SynteticCircleDataset(
        count=100,
        transform=transforms.Compose([
              mtrans.ToLinealMotionBlur(prob=0.8),
              mtrans.ToGaussianBlur(prob=1.0),
              mtrans.ToResizeUNetFoV(388, cv2.BORDER_CONSTANT),
              mtrans.ToTensor()
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
    label = sample_batched['label'][0,1,...]
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
    #plt.imshow( image_b.permute(1,2,0).squeeze() ) 
    plt.imshow( label ) #cmap='gray'
    plt.axis('off')
    plt.ioff()

    plt.subplot(133)
    #plt.imshow( image_c.permute(1,2,0).squeeze()  ) 
    plt.imshow( weight )
    plt.axis('off')

    plt.ioff()       
    plt.show()        

    # observe 4th batch and stop.
    if i_batch == 4: 
        break        

