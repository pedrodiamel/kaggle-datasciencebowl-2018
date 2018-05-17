
import os
import sys
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


sys.path.append('../')
from torchlib.datasets.ctechdata import CTECHDataset  
from torchlib.datasets import imageutl as imutl
from torchlib.datasets import utility as utl
from torchlib.transforms import transforms as mtrans
from torchlib import visualization as view



pathdataset     = '/home/pdmf/.datasets/'
namedataset     = 'cellcaltech0001'
sub_folder      = ''
folders_images  = 'images'
folders_labels  = 'labels'
base_folder = os.path.join(pathdataset, namedataset) 

data = CTECHDataset(
        base_folder, 
        sub_folder, 
        count=10,
        transform=transforms.Compose([
            mtrans.ToResize( (500,500), resize_mode='crop' ) ,
            mtrans.RandomCrop( (255,255), limit=50, padding_mode=cv2.BORDER_CONSTANT  ),
            #mtrans.ToResizeUNetFoV(388, cv2.BORDER_REFLECT_101),
            mtrans.ToTensor(),
            mtrans.ToNormalization(),
        ])
        )


# sample = data[0]
# for k,v in sample.items():
#     print( k, ':', v.shape, v.min(), v.max() )
# print('\n')
# #assert(False)


dataloader = DataLoader(data, batch_size=3, shuffle=False, num_workers=1 )

label_batched = []
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(), sample_batched['label'].size(), )
    
    image = sample_batched['image'][0,:,...]
    label = sample_batched['label'][0,1,...]
    
    print(torch.min(image), torch.max(image), image.shape )
    print(torch.min(label), torch.max(label), image.shape )
    print(image.shape)
    print( np.unique(label) )
        
    plt.figure( figsize=(15,15) )
    plt.subplot(121)
    plt.imshow( image.permute(1,2,0).squeeze()) #, cmap='gray' 
    plt.axis('off')
    plt.ioff()

    plt.subplot(122)
    plt.imshow( label, cmap='gray' ) #
    plt.axis('off')
    plt.ioff()

    # plt.subplot(133)
    # plt.imshow( image_c.permute(1,2,0).squeeze()  ) 
    # #plt.imshow( weight )
    # plt.axis('off')
    #plt.ioff()       
    
    plt.show()        

    # observe 4th batch and stop.
    if i_batch == 2: 
        break        





