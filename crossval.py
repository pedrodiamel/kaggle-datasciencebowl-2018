

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

from sklearn.metrics.pairwise import pairwise_distances


PROJECT='./netruns'
PATHDATASET = '../db'
NAMEDATASET = 'databoewl'
METADATA = 'stage1_train_labels.csv'
PATHMODEL = 'netruns/experiment_unet_dballex_dim388_lr0001_ep1000_adam_b12_wk12_pll_wmce_contours_c0001'
NAMEMODEL = 'chk000205.pth.tar'
METRICSFILE = 'metric.csv'
DEBUG = True  
NUMITER = 10

def iou(gt, pred):
    gt[gt > 0] = 1.
    pred[pred > 0] = 1.
    intersection = gt * pred
    union = gt + pred
    union[union > 0] = 1.
    intersection = np.sum(intersection)
    union = np.sum(union)
    if union == 0:
        union = 1e-09
    return intersection / union


def compute_ious(gt, predictions):
    #gt_ = decompose(gt)
    #predictions_ = decompose(predictions)
    gt_ = gt
    predictions_ = predictions
    
    gt_ = np.asarray([el.flatten() for el in gt_])
    predictions_ = np.asarray([el.flatten() for el in predictions_])
    ious = pairwise_distances(X=gt_, Y=predictions_, metric=iou)
    return ious


def compute_precision_at(ious, threshold):
    mx1 = np.max(ious, axis=0)
    mx2 = np.max(ious, axis=1)
    tp = np.sum(mx2 >= threshold)
    fp = np.sum(mx2 < threshold)
    fn = np.sum(mx1 < threshold)
    return float(tp) / (tp + fp + fn)


def compute_eval_metric(gt, predictions):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ious = compute_ious(gt, predictions)
    precisions = [compute_precision_at(ious, th) for th in thresholds]
    return sum(precisions) / len(precisions)


def pipeline():    
    
    # Configuration
    debug = DEBUG
    pathnamedataset = os.path.join(PATHDATASET, NAMEDATASET);
    pathnamemetadata = os.path.join(PATHDATASET, NAMEDATASET, METADATA)
    pathnamemodel = os.path.join(PATHMODEL, NAMEMODEL)
    
    frec_iter = 1
    base_folder =  pathnamedataset
    sub_folder  =  imutl.trainfile
    folders_image = 'images'
    folders_masks = 'masks'
    id_file_name = METADATA

    segment = proc.Net( ntiles=3 )

    # Load dataset
    print('>> Load dataset ...')

    # load data
    dataloader = imutl.dsxbProvide.create(
        base_folder, 
        sub_folder, 
        id_file_name,
        folders_image, 
        folders_masks,
        )
    print('Total: ', len(dataloader) )


    print('>> Load model ...')
    segment.loadmodel(pathnamemodel)

    print('>> processing ...')

    metrics = list()
    numiter = NUMITER #len(dataloader); 
    for i in range( numiter ):
        
        # load image i
        image, label = dataloader[ i ]

        # segmentation image
        score = segment( image )

        # posprocessing
        labels_est = np.transpose( posp.mpostprocess(score), (1,2,0) )

        if debug:
            labels_color =  labels_est[:,:, np.random.permutation(labels_est.shape[2]) ]
            imagesave = view.makeimagecell(image, labels_color, alphaback=0.7, alphaedge=0.9)
            misc.imsave(os.path.join('netruns/result','{:06d}.png'.format(i)), imagesave )

        # # encode
        # for k in range( labels_est.shape[2] ):      
        #     idimage = dataloader.getid()
        #     rle = nutl.rle_encode( labels_est[:,:,k] > 0 )    
        #     rles.append( {'ImageId':idimage, 'EncodedPixels':rle } )

        y_true = label.transpose( (2,0,1) )
        y_pred = labels_est.transpose( (2,0,1) )

        ious = compute_ious(y_true, y_pred)
        iou_mean = 1.0 * np.sum(ious) / ious.shape[0]
        iout = compute_eval_metric(y_true, y_pred)

        metrics.append( {'IOUMEAN':iou_mean, 'IOUT':iout } )        
        print('>>>', iou_mean, iout)

        if (i+1) % frec_iter == 0:
            print('iteration: {}'.format(i))
    

    return metrics


if __name__ == '__main__':
    
    metrics = pipeline();
    metricsdf = pd.DataFrame(metrics)
    print(metricsdf.describe())

    metircs_filepath = os.path.join(PROJECT, METRICSFILE)
    metricsdf.to_csv(metircs_filepath, index=None, encoding='utf-8')
    
    print('dir: {}'.format(metircs_filepath))
    print('DONE!!!')



