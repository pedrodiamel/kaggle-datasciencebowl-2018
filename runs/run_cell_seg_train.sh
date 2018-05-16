#!/bin/bash

# experiment
# name: exp_[methods]_[arq]_[num]

# parameters
DATA='../db/databoewlex'
PROJECT='./netruns'
EPOCHS=10
BATCHSIZE=1
LEARNING_RATE=0.0001
MOMENTUM=0.5
PRINT_FREQ=10
EXP_NAME='exp_net_unet_008'
WORKERS=1
RESUME='checkpointxx.pth.tar'
GPU=0
ARCH='unet'
LOSS='wmce'
OPT='adam'
SCHEDULER='fixed'
IMAGESIZE=100
SNAPSHOT=10

#rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
#rm -rf $PROJECT/$EXP_NAME/
mkdir $PROJECT    
mkdir $PROJECT/$EXP_NAME  


# --parallel 

python cell_segmentation.py \
$DATA \
--project=$PROJECT \
--name=$EXP_NAME \
--epochs=$EPOCHS \
--batch-size=$BATCHSIZE \
--learning-rate=$LEARNING_RATE \
--momentum=$MOMENTUM \
--image-size=$IMAGESIZE \
--print-freq=$PRINT_FREQ \
--snapshot=$SNAPSHOT \
--workers=$WORKERS \
--resume=$RESUME \
--gpu=$GPU \
--loss=$LOSS \
--opt=$OPT \
--scheduler=$SCHEDULER \
--arch=$ARCH \
--finetuning \
--no-cuda \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \

