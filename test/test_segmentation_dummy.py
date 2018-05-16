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
from torchlib.datasets.syntheticdata import SynteticCircleDataset
from torchlib.neuralnet import SegmentationNeuralNet
from torchlib.datasets import imageutl as imutl
from torchlib.datasets import utility as utl


from torchlib.transforms import transforms as mtrans
from torchlib import visualization as view


project='out/netruns'
name='test_001'
no_cuda=True
parallel=False
seed=1
print_freq=10
gpu=0
arch='unet'
num_classes=3
num_channels=3 
loss='mcedice'
lr=0.0001
momentum=0.99
opt='adam'
scheduler='fixed'
finetuning=False
nepoch=5
size_input=284

network = SegmentationNeuralNet(
        patchproject=project,
        nameproject=name,
        no_cuda=no_cuda,
        parallel=parallel,
        seed=seed,
        print_freq=print_freq,
        gpu=gpu
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

print(network)

data = SynteticCircleDataset(
        count=100,
        imsize=(250,250),
        sigma=0.01,
        bdraw_grid=False,
        transform=transforms.Compose([
              mtrans.ToResizeUNetFoV(size_input, cv2.BORDER_CONSTANT),
              mtrans.ToRandomTransform( mtrans.ToGaussianBlur(), prob=0.5 ),                                         
              mtrans.ToTensor(),
              mtrans.ToNormalization(),
            ])
        )

sample = data[0]
for k,v in sample.items():
    print( k, ':', v.shape, v.min(), v.max() )

dataloader_train = DataLoader(data, batch_size=3, shuffle=False, num_workers=1 )
dataloader_val = DataLoader(data, batch_size=3, shuffle=False, num_workers=1 )
dataloader_test = DataLoader(data, batch_size=3, shuffle=False, num_workers=1 )


network.evaluate(dataloader_train, epoch=0)

for epoch in range(nepoch):
    
    print('\nEpoch: {}/{} ({}%)'.format(epoch, nepoch, int((float(epoch)/nepoch)*100) ) )
    print('-' * 25)
    
    network.training(dataloader_val, epoch=epoch)
    network.evaluate(dataloader_test, epoch=epoch+1)

print('DONE!!!')


