
import os
import numpy as np
from skimage import color
import scipy.misc
from scipy import ndimage as ndi
import cv2

from deep.datasets import imageutl as imutl
from deep.datasets import utility as utl
from deep.datasets import weightmaps 
from deep import preprocessing as prep
from deep import netutility as netutl

from skimage import io, transform, morphology, filters
import skimage.morphology as morph
import skfmm

from deep.datasets import weightmaps 


imsize = 250
pcval = 10 #%
numiter = 10


def save(i, pathname, 
    image_t, label_t, 
    contours_t,
    centers_t,
    touchs_t,
    weight_t, 
    prefijo=''):   

    scipy.misc.imsave(os.path.join(pathname, 'images'   ,'{}{:06d}.png'.format(prefijo,i)), image_t )
    scipy.misc.imsave(os.path.join(pathname, 'labels'   ,'{}{:06d}.png'.format(prefijo,i)), label_t )
    scipy.misc.imsave(os.path.join(pathname, 'contours' ,'{}{:06d}.png'.format(prefijo,i)), contours_t )
    scipy.misc.imsave(os.path.join(pathname, 'centers'  ,'{}{:06d}.png'.format(prefijo,i)), centers_t )
    scipy.misc.imsave(os.path.join(pathname, 'touchs'   ,'{}{:06d}.png'.format(prefijo,i)), touchs_t )

    np.savetxt(os.path.join(pathname, 'weights', '{}{:06d}.txt'.format(prefijo,i)), weight_t, fmt="%2.3f", delimiter=",")
    print('>>', os.path.join(pathname, '{}{:06d}'.format(prefijo, i)) )



def tolabel(mask):
    labeled, nr_true = ndi.label(mask)
    return np.array(labeled)

def decompose(labeled):
    nr_true = labeled.max()
    masks = []
    for i in range(1, nr_true + 1):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        masks.append(msk)
    if not masks: return np.array([labeled])
    else: return np.array(masks)

def preprocessing(image, label, imsize=250, bcrop=False):
    

    if bcrop:
        barea = False
        while barea == False:     
            image_t, label_t = prep.randomcrop(image, label, (imsize,imsize) )  
            bcontour = (label_t==128)
            label_t = tolabel(label_t>128)
            label_t = decompose(label_t).transpose( (1,2,0) )
            masks = (label_t.transpose((2,0,1))>0).astype(np.uint32)
            masks = np.array([ ndi.morphology.binary_fill_holes(x) for x in masks ])
            masks = prep.delete_black_layer(masks)
            barea = np.sum(masks) > 0   
        image = image_t
        label = label_t

    else:      

        bcontour = (label_t==128)
        label = decompose(tolabel(label>128)).transpose( (1,2,0) )
        masks = (label.transpose((2,0,1))>0).astype(np.uint32)
        masks = np.array([ ndi.morphology.binary_fill_holes(x) for x in masks ])
        
    bmask = (np.max(masks,axis=0)>0)

    # preprocessing
    bmask, bcontour, btouch, bcenters = prep.create_groundtruth(masks)
    weight   = weightmaps.getunetweightmap( bmask + 2*btouch, masks, w0=10, sigma=5, )

    #btouch   = np.zeros_like( bmask )
    #bcenters = np.zeros_like( bmask )

    return image, bmask, bcontour, btouch, bcenters, weight

def create( dataloader, numiter, pcval, bcrop ):
    
    for i in range( numiter ):        
        k = i%(len(dataloader))
        image, label = dataloader[ i ]


        minsize = np.min(image.shape[:2])
        if  minsize < imsize:
            asp = imsize/minsize
            print('>>>>>', image.shape)
            image = cv2.resize(image, None, fx=2*asp, fy=2*asp , interpolation = cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=2*asp, fy=2*asp , interpolation = cv2.INTER_LINEAR)

        # ini = 100
        # timsize = 300
        # image = image[ini:ini+timsize,ini:ini+timsize,:]
        # label = label[ini:ini+timsize,ini:ini+timsize]

        print('>> ', dataloader.getid() )  
        
        image_t, bmask_t, bcontour_t, btouch_t, bcenters_t, weight_t = preprocessing(image, label, imsize, bcrop=bcrop)

        save( 
            i, 
            os.path.join(pathnameout,'train' if i < numiter - int(pcval*numiter/100) else 'test' ), 
            image_t, bmask_t, bcontour_t, bcenters_t, btouch_t, weight_t, 
            'n' 
            )  
    


if __name__ == '__main__':

    pathdataset = '../db/dbselect'
    namedataset = 'nc001'
    pathname = os.path.join(pathdataset, namedataset);
    pathnameout = '../db'
    namedatasetout = 'databoewlex'
    pathnameout = os.path.join(pathnameout, namedatasetout)
    folders_images  = 'images'
    folders_labels  = 'labels'
    sub_folder = ''

    print('>> ', 'Create dir: ',  pathnameout)
    if os.path.exists(pathnameout) is not True:
        os.makedirs(pathnameout);
        os.makedirs(os.path.join(pathnameout,'train'));
        os.makedirs(os.path.join(pathnameout,'test'));
        #train
        os.makedirs(os.path.join(pathnameout,'train','images'));
        os.makedirs(os.path.join(pathnameout,'train','labels'));
        os.makedirs(os.path.join(pathnameout,'train','weights'));
        os.makedirs(os.path.join(pathnameout,'train','contours'));
        os.makedirs(os.path.join(pathnameout,'train','centers'));
        os.makedirs(os.path.join(pathnameout,'train','touchs'));
        #test
        os.makedirs(os.path.join(pathnameout,'test','images'));
        os.makedirs(os.path.join(pathnameout,'test','labels'));
        os.makedirs(os.path.join(pathnameout,'test','weights'));
        os.makedirs(os.path.join(pathnameout,'test','contours'));
        os.makedirs(os.path.join(pathnameout,'test','centers'));
        os.makedirs(os.path.join(pathnameout,'test','touchs'));

    print('>> ', 'loading dataset ...')
    dataloader = imutl.nucleiProvide.create(
        pathname, 
        sub_folder, 
        folders_images, 
        folders_labels,
        )

    print('>> ','loader ok :)!!!')
    print('>> ',len(dataloader))
    
    create( dataloader, numiter, pcval, True )

    print('Done!!!!')
