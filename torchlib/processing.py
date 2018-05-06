
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as nnfun

import cv2
import numpy as np

import deep
from deep import netmodels as nnmodels
from deep.datasets import dsxbtransform as dsxbtrans

from skimage import color
import scipy.misc
import math

def unet_transform_image_size(image, size=388):
    
    height, width, channels = image.shape;
    image = np.array(image)
    
    asp = float(height)/width;
    w = size; 
    h = int(w*asp);
    
    #image_x = scipy.misc.imresize(image.copy(), (h,w), interp='bilinear');
    image_x = cv2.resize(image, (w,h) , interpolation = cv2.INTER_CUBIC)

    image = np.zeros((w,w,3));
    ini = int(round((w-h) / 2.0));
    image[ini:ini+h,:,:] = image_x;

    #image = image/255.0
    
    downsampleFactor = 16;
    d4a_size= 0;
    padInput =   (((d4a_size *2 +2 +2)*2 +2 +2)*2 +2 +2)*2 +2 +2;
    padOutput = ((((d4a_size -2 -2)*2-2 -2)*2-2 -2)*2-2 -2)*2-2 -2;
    d4a_size = math.ceil( (size - padOutput)/downsampleFactor);
    input_size = downsampleFactor*d4a_size + padInput;
    offset=(input_size-size)//2
    image_f = np.zeros((input_size,input_size,3));
    image_f[offset:-offset,offset:-offset,:]=image;

    
    return image_f, asp, image;


def inv_transform_image_size(image, imshape, asp):
    '''
    Invert transform image size
    '''
    
    height, width = image.shape[:2];
    image = np.array(image);    
    w = height; 
    h = int(width*(asp));    
    ini = int(round((w-h) / 2.0));
    image_x = image[ini:h+ini,:,:];
    #image_x = scipy.misc.imresize(image_x, imshape, interp='bilinear', mode='F');
    image_x = cv2.resize(image_x, (imshape[1],imshape[0]) , interpolation = cv2.INTER_CUBIC)

    return image_x;


def create_input(data, fovnet , mode = 'mirror'):
    
    mH,mW = data.shape[:2]
    h,w = fovnet[:2]
    cx = mW//w; padx = (w - mW%w)   
    cy = mH//h; pady = (h - mH%h)   
    
    border = pady, padx
    borderL = border[0]//2 
    borderR = np.ceil(border[0]/2).astype(np.uint8)     
    borderT = border[1]//2 
    borderB = np.ceil(border[1]/2).astype(np.uint8)
        
    paddedFullImage = np.zeros( (data.shape[0] + border[0], data.shape[1] + border[1], data.shape[2] ) );
    paddedFullImage[ borderL:(borderL+data.shape[0]), borderT:(borderT+data.shape[1]), : ] = data;
    
    if mode == 'mirror':
        
        xpadL  = borderL;
        xfromL = borderL+1;
        xtoL   = borderL+data.shape[0];         
        xpadR  = borderR;
        xfromR = borderR+1;
        xtoR   = borderR+data.shape[0];
        
        paddedFullImage[:xfromL,:,:] = paddedFullImage[ xfromL-1:xfromL+xpadL,:,:][::-1,:,:];
        paddedFullImage[xtoL:,:,:] = paddedFullImage[xtoL-xpadR:xtoL, :,:][::-1,:,:] ;

        ypadT  = borderT;
        yfromT = borderT+1;
        ytoT   = borderT+ data.shape[1];        
        ypadB  = borderB;
        yfromB = borderB+1;
        ytoB   = borderB+ data.shape[1];
        
        paddedFullImage[:, :yfromT,:] = paddedFullImage[ :, yfromT-1:yfromT+ypadT,:][:,::-1,:];
        paddedFullImage[:, ytoT:,:] = paddedFullImage[ :, ytoT-ypadB:ytoT, :][:,::-1,:];
    
    return paddedFullImage, border
    

def sliding_windows( data, sizewnd=(572,572) ):
    h,w = data.shape[:2]
    ny, nx = h//sizewnd[0], w//sizewnd[1]    
    for i in range(ny):
        for j in range(nx):
            inix = j*sizewnd[1] ;iniy = i*sizewnd[0]
            yield( iniy, inix, data[iniy:iniy+sizewnd[0], inix:inix+sizewnd[1],:] )


def center_crop(image, newsize ):            
    height, width = newsize
    h, w, c = image.shape
    dy = (h - height) // 2
    dx = (w - width)  // 2
    y1 = dy; y2 = y1 + height
    x1 = dx; x2 = x1 + width
    image_t  =  image[y1:y2, x1:x2, :]
    return image_t


class Net(object):
                

    def __init__(self, average_mirror=True ): 
                
        self.bcuda = torch.cuda.is_available()
        
        d4a_size = 0
        # self.n_tilesx = ntilesx
        # self.n_tilesy = ntilesy
        self.padOutput = ((((d4a_size -2 -2)*2-2 -2)*2-2 -2)*2-2 -2)*2-2 -2
        self.padInput  = (((d4a_size*2 +2 +2)*2 +2 +2)*2 +2 +2)*2 +2 +2
        self.downsampleFactor = 16
        self.padding = 'mirror'
        self.average_mirror = average_mirror    
                            
        
    def loadmodel(self, pathnamemodel):
        
        self.pathnamemodel = pathnamemodel
        if self.bcuda: model = torch.load( pathnamemodel )
        else: model = torch.load( pathnamemodel, map_location=lambda storage, loc: storage )
        
        self.num_classes = model['num_classes']
        self.arch = model['arch']

        if self.arch == 'unet':
            self.net = nnmodels.unet( n_classes = self.num_classes, in_channels=3 )  
        elif self.arch == 'dunet':
            self.net = nnmodels.dunet( n_classes = self.num_classes, in_channels=3)   
        elif self.arch == 'unet11':
            self.net = nnmodels.unet11( num_classes = self.num_classes, in_channels=3 ) 
        else:
            assert(False)

        if self.bcuda: self.net.cuda()
        self.net.load_state_dict( model['state_dict'] )
        self.net.eval()
        
        print('>> Model loader ready ...')
        
    
    def __call__(self, image):
        
        #preprocessing 
        h,w = image.shape[:2]
        if w<h: image = np.transpose( image, (1,0,2) )  
        
        nh,nw = image.shape[:2]
        asp = float(nh)/nw
        rw = 388
        rh = int(rw*asp)

        #image = cv2.fastNlMeansDenoisingColored(image,None,5,5,7,21)
        tranfClahe = dsxbtrans.CLAHE()
        image = tranfClahe( image ) 
        
        image = cv2.resize(image, None, fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
        #image = cv2.resize(image, (rw,rh), interpolation = cv2.INTER_CUBIC)     
        #image = cv2.GaussianBlur(image, (3, 3), 0);
        #image = cv2.bilateralFilter(image,9,75,75)   

        image = image.astype( np.float32 )
        for i in range( image.shape[2] ):        
           image[:,:,i] = image[:,:,i] - image[:,:,i].min()
           image[:,:,i] = image[:,:,i] / image[:,:,i].max()

        #fit title
        
        #score = self._sliding_forward(image, sizewnd= (388,388))
        # image_in, asp, _ = unet_transform_image_size(image, size=512)
        # score = self._forward( image_in )
        # score = inv_transform_image_size(score, (nh,nw), asp)

       
        n_tilesx = 1 #image.shape[0]//128
        n_tilesy = 2 #image.shape[1]//128
        score = self._tiled_forward(image, n_tilesx=n_tilesx, n_tilesy=n_tilesy)        
        score = cv2.resize(score, (nw,nh), interpolation = cv2.INTER_CUBIC)

        if w<h: score = np.transpose( score, (1,0,2) )

        return score
       
    
    def _sliding_forward( self, image, sizewnd=(388,388), average_mirror=True ):
        
        #imsize = image.shape[:2]
        imsize = sizewnd

        d4a_size = 0
        downsampleFactor = 16,
        padInput = (((d4a_size*2 +2 +2)*2 +2 +2)*2 +2 +2)*2 +2 +2,
        padOutput = ((((d4a_size -2 -2)*2-2 -2)*2-2 -2)*2-2 -2)*2-2 -2,

        d4a_size = np.ceil( (np.array([imsize[0], imsize[1]]) - padOutput)/downsampleFactor  )    
        input_size  = (downsampleFactor*d4a_size + padInput).astype(int);
        output_size = (downsampleFactor*d4a_size + padOutput).astype(int); 

        #print(d4a_size)
        #print(input_size)
        #print(output_size)

        image_t, border = create_input( image, input_size )
        #print(image_t.shape)

        nClasses = self.num_classes
        image_r = np.zeros( ( (image_t.shape[0]//input_size[0])*output_size[0] , (image_t.shape[1]//input_size[0])*output_size[1] , nClasses) )

        for (i,j, paddedInputSlice ) in sliding_windows( image_t, input_size ):
            print(i,j, paddedInputSlice.shape)            
            
            scoreSlice = self._forward( paddedInputSlice )      
            if average_mirror:            
                scores_torch = self._forward(np.fliplr(paddedInputSlice))
                scoreSlice   = np.fliplr(scores_torch)
                scores_torch = self._forward(np.fliplr(paddedInputSlice))
                scoreSlice   = np.fliplr(scores_torch)
                scores_torch = self._forward(np.flipud(paddedInputSlice))
                scoreSlice   = np.flipud(scores_torch)
                scores_torch = self._forward(np.flipud(np.fliplr(paddedInputSlice)) )
                scoreSlice   = np.flipud(np.fliplr(scores_torch))
                scoreSlice   = scoreSlice/4;
            
            #scoreSlice = np.zeros( (388,388,2) )
            
            print(i,j,nClasses)
            print(scoreSlice.shape)
            print(output_size)
            print(image_r.shape)

            image_r[i:(i+1)*output_size[0],j:(j+1)*output_size[1],:] = scoreSlice


        #image_r = image_r[ (border[0]//2):(border[0]//2)+image.shape[0], (border[1]//2):(border[1]//2) + image.shape[1], : ]    
        image_r = center_crop(image_r, image.shape[:2] )
        print(image_r.shape)
        
        return image_r
   
        
    def _tiled_forward( self, data, n_tilesx=2, n_tilesy=2 ):
    
        # n_tilesx = self.n_tilesx
        # n_tilesy = self.n_tilesy

        padOutput = self.padOutput
        padInput  = self.padInput
        downsampleFactor = self.downsampleFactor
        padding = self.padding
        average_mirror = self.average_mirror

        imsize = np.array(data.shape[:2])
        d4a_size = np.ceil( (np.array([np.ceil( imsize[0]/n_tilesx ) , np.ceil( imsize[1]/n_tilesy ) ]) - padOutput)/downsampleFactor  )    
   
        input_size = downsampleFactor*d4a_size + padInput;
        output_size = downsampleFactor*d4a_size + padOutput;        
        border = (np.round(input_size-output_size)//2);

        border = border.astype(int)
        input_size = input_size.astype(int)
        output_size = output_size.astype(int)

        paddedFullImage = np.zeros( (data.shape[0] + 2*border[0], data.shape[1] + 2*border[1], data.shape[2] ) );
        paddedFullImage[ border[0]:border[0]+data.shape[0], border[1]:border[1]+data.shape[1], : ] = data;

        if padding == 'mirror':
            xpad  = border[0];
            xfrom = border[0]+1;
            xto   = border[0]+data.shape[0];        
            paddedFullImage[:xfrom,:,:] = paddedFullImage[ xfrom-1:xfrom+xpad,:,:][::-1,:,:];
            paddedFullImage[xto:,:,:] = paddedFullImage[xto-xpad:xto, :,:][::-1,:,:] ;
            ypad  = border[1];
            yfrom = border[1]+1;
            yto   = border[1]+ data.shape[1];        
            paddedFullImage[:, :yfrom,:] = paddedFullImage[ :, yfrom-1:yfrom+ypad,:][:,::-1,:];
            paddedFullImage[:, yto:,:] = paddedFullImage[ :, yto-ypad:yto, :][:,::-1,:];

        nClasses = self.num_classes
        scores = np.zeros( (data.shape[0],data.shape[1], nClasses) );
        for yi in range(n_tilesy):
            for xi in range(n_tilesx):

                paddedInputSlice = np.zeros( (input_size[0], input_size[1], data.shape[2]) )
                validReg_x = min( input_size[0], paddedFullImage.shape[0] - xi*output_size[0] )
                validReg_y = min( input_size[1], paddedFullImage.shape[1] - yi*output_size[1] )
                paddedInputSlice[:validReg_x, :validReg_y] = paddedFullImage[xi*output_size[0]:xi*output_size[0]+validReg_x, yi*output_size[1]:yi*output_size[1]+validReg_y,:]

                scoreSlice = self._forward( paddedInputSlice )      
                if average_mirror:            
                    scores_torch = self._forward(np.fliplr(paddedInputSlice))
                    scoreSlice   = np.fliplr(scores_torch)
                    scores_torch = self._forward(np.fliplr(paddedInputSlice))
                    scoreSlice   = np.fliplr(scores_torch)
                    scores_torch = self._forward(np.flipud(paddedInputSlice))
                    scoreSlice   = np.flipud(scores_torch)
                    scores_torch = self._forward(np.flipud(np.fliplr(paddedInputSlice)) )
                    scoreSlice   = np.flipud(np.fliplr(scores_torch))
                    scoreSlice   = scoreSlice/4;

                print('>>')

                validReg_x = min(output_size[0], scores.shape[0] - xi*output_size[0] );
                validReg_y = min(output_size[1], scores.shape[1] - yi*output_size[1] );        
                scores[xi*output_size[0]:xi*output_size[0]+validReg_x, 
                    yi*output_size[1]:yi*output_size[1]+validReg_y,:] = scoreSlice[:validReg_x,:validReg_y,:];

        return scores
        
    

    def _forward( self, image_in ):

        image_proc = image_in[:, :, :]
        #image_proc = cv2.cvtColor( (image_proc*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        #image_proc  = image_proc.mean(axis=2)
        #image_proc = image_proc[:,:,np.newaxis]

        image_proc = image_proc.astype(np.float32)

        # NHWC -> NCHW
        image_proc = image_proc.transpose(2, 0, 1)
        image_proc = image_proc[np.newaxis,...]
        image_proc = torch.from_numpy(image_proc).float()


        if self.bcuda:
            images_torch = Variable(image_proc.cuda(0), volatile=True )
        else:
            images_torch = Variable(image_proc)

        # fordward net
        score = self.net(images_torch)
        score = score.data.cpu().numpy().transpose(2,3,1,0)[...,0]   
        return score
    