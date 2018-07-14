#!/bin/bash

PATHDATASET='~/.datasets/datasciencebowl'
NAMEDATASET='databoewl'
PATHNAMEOUT='.'
FILENAME='submission.csv'
PATHMODEL='netruns/experiment_unet_fx_c0001'
NAMEMODEL='chk000255.pth.tar'
MODEL=$PATHMODEL/$NAMEMODEL  


python submission.py \
--pathdataset=$PATHDATASET \
--namedataset=$NAMEDATASET \
--pathnameout=$PATHNAMEOUT \
--filename=$FILENAME \
--model=$MODEL \


