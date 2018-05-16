#!/bin/bash

PATHDATASET='../db'
NAMEDATASET='databoewl'
METADATA='stage1_train_labels.csv'
PATHNAMEOUT='../db'
NAMEDATASETOUT='databoewlex'
SIZECROP=512
PERCENTTEST=10 #%
SIZEDATASET=10

python create_dataset.py \
--pathdataset=$PATHDATASET \
--namedataset=$NAMEDATASET \
--pathnameout=$PATHNAMEOUT \
--namedatasetout=$NAMEDATASETOUT \
--metadata=$METADATA \
--percent-test=$PERCENTTEST \
--sizecrop=$SIZECROP \
--sizedataset=$SIZEDATASET \


