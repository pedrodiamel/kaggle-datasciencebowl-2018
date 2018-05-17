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
from torchlib.datasets.syntheticdata import SyntethicCircleDataset
from torchlib.neuralnet import SegmentationNeuralNet
from torchlib.datasets import imageutl as imutl
from torchlib.datasets import utility as utl

from torchlib.transforms import transforms as mtrans
from torchlib import visualization as view
from torchlib.logger import summary


project='../out/netruns'
name='test_001'
no_cuda=False
parallel=True
seed=1
print_freq=10
gpu=0
arch='unet'
num_classes=3
num_channels=3 
loss='wmce'
lr=0.0001
momentum=0.99
opt='adam'
scheduler='fixed'
finetuning=False
nepoch=10
size_input=388
snapshot=5
view_freq=1
num_workers=10
batch_size=10
count=1000

network = SegmentationNeuralNet(
        patchproject=project,
        nameproject=name,
        no_cuda=no_cuda,
        parallel=parallel,
        seed=seed,
        print_freq=print_freq,
        gpu=gpu,
        view_freq=view_freq,
        )

network.create(
        arch=arch, 
        num_output_channels=num_classes, 
        num_input_channels=num_channels,  
        loss=loss, 
        lr=lr, 
        momentum=momentum,
        optimizer=opt,
        lrsch=scheduler,
        pretrained=finetuning,
        size_input=size_input
        )

#print(network)
#summary(network.net, [num_classes,size_input,size_input] )
print('load model ...')

data = SyntethicCircleDataset(
        count=count,
        imsize=(250,250),
        sigma=0.01,
        bdraw_grid=False,
        generate=SyntethicCircleDataset.generate_image_mask_and_weight,
        transform=transforms.Compose([
              mtrans.ToResizeUNetFoV(size_input, cv2.BORDER_CONSTANT),
              mtrans.ToRandomTransform( mtrans.ToGaussianBlur(), prob=0.5 ),                                         
              mtrans.ToTensor(),
              mtrans.RandomElasticTensorDistort( size_grid=10, deform=0.05 ),
              mtrans.ToNormalization(),
            
            ])
        )

print('load dataset ...')
print('Size dataset:', len(data))
#sample = data[0]
#for k,v in sample.items():
#    print( k, ':', v.shape, v.min(), v.max() )
#print('\n')
#assert(False)

train_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers )
val_loader   = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers )
test_loader  = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers )

# training neural net
network.fit( train_loader, val_loader, nepoch, snapshot )

#network.evaluate(val_loader, epoch=0)
#for epoch in range(nepoch):    
#    print('\nEpoch: {}/{} ({}%)'.format(epoch, nepoch, int((float(epoch)/nepoch)*100) ) )
#    print('-' * 25)
#    network.training(val_loader, epoch=epoch)
#    network.evaluate(train_loader, epoch=epoch+1)

print('DONE!!!')


