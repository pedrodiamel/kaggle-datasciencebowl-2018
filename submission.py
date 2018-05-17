
import os
import sys
import argparse
import numpy as np
import pandas as pd

from deep.datasets import imageutl as imutl
from deep.datasets import utility as utl
from deep.datasets import weightmaps 
from deep import netmodels as nnmodels
from deep import visualization as view
from deep import netutility as nutl
from deep import neuralnet as deepnet
from deep import postprocessing as posp
from deep import processing as proc


from skimage.color import label2rgb
from skimage import measure
from skimage import morphology
import scipy.misc as misc

from argparse import ArgumentParser


def arg_parser():
    """Arg parser"""    
    parser = ArgumentParser()
    parser.add_argument('--pathdataset', metavar='DIR',  help='path to dataset')
    parser.add_argument('--namedataset', metavar='S',  help='name to dataset')
    parser.add_argument('--pathnameout', metavar='DIR',  help='path to out dataset')
    parser.add_argument('--filename', metavar='S', help='name of the file output')
    parser.add_argument('--model', metavar='S',  help='filename model')  
    return parser


def pipeline( dataloader, segment, frec_iter=10):    
    

    rles = list()
    numiter = len(dataloader); 
    for i in range( numiter ):
        
        # load image i
        image = dataloader[ i ]

        # segmentation image
        score = segment( image )

        # posprocessing
        labels_est = np.transpose( posp.mpostprocess(score), (1,2,0) )

        # encode
        for k in range( labels_est.shape[2] ):      
            idimage = dataloader.getid()
            rle = nutl.rle_encode( labels_est[:,:,k] > 0 )    
            rles.append( {'ImageId':idimage, 'EncodedPixels':rle } )

        if (i+1) % frec_iter == 0:
            print('iteration: {}'.format(i))
        

    return rles


if __name__ == '__main__':
    
    parser = arg_parser();
    args = parser.parse_args();

    # Configuration
    pathnamedataset = os.path.join(args.pathdataset, args.namedataset);
    pathnamemodel =  args.model
    pathnameout  = args.pathnameout
    filename = args.filename

    n_tiles = 3
    frec_iter = 1
    base_folder =  pathnamedataset
    sub_folder  =  imutl.testfile
    folders_image = 'images'  

    # Load dataset
    print('>> Load dataset ...')

    # load data
    dataloader = imutl.dsxbImageProvide.create(
        base_folder, 
        sub_folder, 
        folders_image, 
        )
    print('Total: ', len(dataloader) )


    print('>> Load model ...')
    segment = proc.Net( ntiles=n_tiles  )
    segment.loadmodel(pathnamemodel)


    print('>> processing ...')    
    rles = pipeline(dataloader, segment, frec_iter);
    submission = pd.DataFrame(rles).astype(str)
    submission = submission[submission['EncodedPixels']!='nan']
    submission_filepath = os.path.join(pathnameout, filename)
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    
    print('dir: {}'.format(submission_filepath))
    print('DONE!!!')



