#!/bin/bash


# parameters
DATA='~/.datasets/datasciencebowl/databoewlex'
NAMEDATASET='databoewlex'
PROJECT='../netruns'
EPOCHS=60
BATCHSIZE=60
LEARNING_RATE=0.0001
MOMENTUM=0.5
PRINT_FREQ=50
WORKERS=60
RESUME='checkpointxx.pth.tar'
GPU=0
ARCH='unet'
LOSS='wmce'
OPT='adam'
SCHEDULER='fixed'
IMAGESIZE=100
SNAPSHOT=10
EXP_NAME='exp_datasciencebowl_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_001'


rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
rm -rf $PROJECT/$EXP_NAME/
mkdir $PROJECT    
mkdir $PROJECT/$EXP_NAME  


python ../train.py \
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
--parallel \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \

