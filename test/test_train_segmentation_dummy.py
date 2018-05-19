import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from pytvision.datasets.syntheticdata import SyntethicCircleDataset
from pytvision.transforms import transforms as mtrans
from pytvision import visualization as view

sys.path.append('../')
from torchlib.neuralnet import SegmentationNeuralNet


project='../out/netruns'
name='test_001'
no_cuda=True
parallel=False
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
size_input=100
snapshot=5
view_freq=1
num_workers=1
batch_size=3
count=100


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

print('||> create model ...')
start = time.time()
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

t = time.time() - start
print('||> create model time: {}sec'.format(t) )

#print(network)

print('||> load dataset ...')

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


print('||> size dataset:', len(data))
#sample = data[0]
#for k,v in sample.items():
#    print( k, ':', v.shape, v.min(), v.max() )
#print('\n')
#assert(False)

train_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers )
val_loader   = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers )
test_loader  = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers )

# training neural net
start = time.time()
network.fit( train_loader, val_loader, nepoch, snapshot )
t = time.time() - start

print('||> fit model time: {}sec'.format(t) )
print('||> DONE!!!')


