
import os
from skimage import io, transform
import numpy as np
from skimage import color
import scipy.misc

from torchlib.datasets import imageutl as imutl
from torchlib.datasets import utility as utl
from torchlib.datasets import weightmaps 
from torchlib import preprocessing as prep

from argparse import ArgumentParser
import datetime


def arg_parser():
    """Arg parser"""    
    parser = ArgumentParser()
    parser.add_argument('--pathdataset', metavar='DIR',  help='path to dataset')
    parser.add_argument('--namedataset', metavar='S',  help='name to dataset')
    parser.add_argument('--pathnameout', metavar='DIR',  help='path to out dataset')
    parser.add_argument('--namedatasetout', metavar='S',  help='name to out dataset')
    parser.add_argument('--metadata', metavar='S',  help='metadata file')
    parser.add_argument('--sizedataset', default=200, type=int, metavar='N', help='size dataset')
    parser.add_argument('--percent-test', default=10, type=int, metavar='N', help='percent set test')
    parser.add_argument('--sizecrop', default=512, type=int, metavar='N', help='size crop')
    return parser


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


def create( dataloader, files, numiter, pcval, bcrop, imsize ):
    
    for i in range( numiter ):
        
        k = i%(len(files))
        image, label = dataloader[ files[ k ] ]
        #image, label = dataloader[ i ]
        print( dataloader.getid() )  
        
        image_t, bmask_t, bcontour_t, btouch_t, bcenters_t, weight_t = prep.preprocessing(image, label, imsize, bcrop=bcrop)

        save( 
            i, 
            os.path.join(pathnameout,'train' if i < numiter - int(pcval*numiter/100) else 'test' ), 
            image_t, 
            bmask_t,
            bcontour_t, 
            bcenters_t,
            btouch_t,
            weight_t, 
            grups[g] 
            )  
    


if __name__ == '__main__':
    
    # parameters
    parser = arg_parser();
    args = parser.parse_args();

    pathdataset = args.pathdataset
    namedataset = args.namedataset
    metadata = args.metadata
    pathname = os.path.join(pathdataset, namedataset);
    pathmetadata = os.path.join(pathdataset, namedataset, metadata)
    pathnameout = args.pathnameout 
    namedatasetout = args.namedatasetout
    pathnameout = os.path.join(pathnameout, namedatasetout)

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

    print('Create dir: ',  pathnameout)

    dataloader = imutl.dsxbProvide.create(
        base_folder=pathname, 
        sub_folder=imutl.trainfile, 
        id_file_name=metadata, 
        folders_image='images', 
        folders_masks='masks',
        )

    print(len(dataloader))
    print('loader ok :)!!!')
    
    grups  = ['a','b','c','d','e','f','g']
    bcrops = [True, False, True, False, False, False, False]

    for g in range( len(grups)  ):

        files = np.loadtxt('grup_{}.txt'.format(grups[g]), delimiter=",")
        files = np.array(files).astype(np.uint32)

        print('Select files')
        print(files)

        create( dataloader, files, args.sizedataset, args.percent_test, bcrops[g], args.sizecrop )


    print('Done!!!!')
