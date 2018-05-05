
import os
from skimage import io, transform
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


imsize = 512
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

def create( dataloader, numiter, pcval, bcrop ):
    
    for i in range( numiter ):        
        k = i%(len(dataloader))
        image, label = dataloader[ k ]

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        label = decompose(tolabel(label)).transpose( (1,2,0) )
        print('>> ', dataloader.getid() )  
        
        image_t, bmask_t, bcontour_t, btouch_t, bcenters_t, weight_t = prep.preprocessing(image, label, imsize, bcrop=bcrop)
        save( 
            i, 
            os.path.join(pathnameout,'train' if i < numiter - int(pcval*numiter/100) else 'test' ), 
            image_t, bmask_t, bcontour_t, bcenters_t, btouch_t, weight_t, 
            'k' 
            )  
    


if __name__ == '__main__':

    pathdataset = '../db/dbselect'
    namedataset = 'kt001'
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
    dataloader = imutl.ctechProvide.create(
        pathname, 
        sub_folder, 
        folders_images, 
        folders_labels,
        )

    print('>> ','loader ok :)!!!')
    print('>> ',len(dataloader))
    
    create( dataloader, numiter, pcval, True )

    print('Done!!!!')
