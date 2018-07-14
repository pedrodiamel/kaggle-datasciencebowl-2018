
import os
import sys

import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import scipy.misc
import cv2

from pytvision.datasets.syntheticdata import SyntethicCircleDataset
from pytvision.transforms import transforms as mtrans
from pytvision import visualization as view

sys.path.append('../')
from torchlib.segneuralnet import SegmentationNeuralNet
from torchlib import postprocessing as posp


project='../out/netruns'
name='test_001'
namemodel='model_best.pth.tar'
path_model = os.path.join(project, name, 'models', namemodel )

no_cuda=False
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

# load model
print('||> load model ...')
start = time.time()
if network.load( path_model ) is not True:
    assert(False)
t = time.time() - start
print('||> load model time: {}sec'.format(t) )


print('||> load dataset ...')
data = SyntethicCircleDataset(
        count=count,
        imsize=(500,500),
        sigma=0.01,
        bdraw_grid=True,
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


# training neural net
sample = data[0]
image, label = sample['image'], sample['label'] 
image = image.unsqueeze(0)
print(image.shape)

score = network.inference( image )

print(score.shape)
print('||> save score ...')
for c in range(score.shape[2]):
    scipy.misc.imsave('../out/score_{}.png'.format(c), score[:,:,c])

#label = posp.mpostprocessthresh(score, prob=0.5)
#label = posp.mpostprocessmax(score )
label = posp.mpostprocess(score)

print(label.shape)

# save result
print('||> save image and result ...')
im = image.data.cpu().numpy().transpose(2, 3, 1,0)[:,:,:,0]
print(im.shape)
scipy.misc.imsave('../out/image.png', im )
scipy.misc.imsave('../out/result.png', (label/label.max())*255 )


print('||> DONE!!!')


