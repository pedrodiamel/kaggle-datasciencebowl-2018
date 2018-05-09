
import torch
import numpy as np
import cv2

from . import utility as utl
from . import functional as F

class ObjectTransform(object):
    def __init__(self, image ):
        self.image = image
    
    #blur transforms
    
    ### lineal blur transform
    def lineal_blur(self, gen):        
        self.image, reg = gen.generatelineal( self.image ) 
    
    ### motion blur transform
    def motion_blur(self, gen):        
        self.image, reg = gen.generate( self.image ) 

    ### gaussian blur
    def gaussian_blur(self, wnd):
        self.image = cv2.GaussianBlur(self.image, (wnd, wnd), 0); 

    #colors transforms

    ### add noice
    def add_noise(self, noise):
        
        image = self.image
        assert( np.any( image.shape[:2] == noise.shape ) )

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        gray, a, b = cv2.split(lab)    
        gray = gray.astype(np.float32)/255
        
        H,W  = gray.shape
        noisy = gray + noise
        noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)

        lab   = cv2.merge((noisy, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        self.image = image

    ### brightness
    def brightness(self, alpha):
        img = self.image
        maxval = np.max(img[..., :3])
        dtype = img.dtype
        img[..., :3] = utl.clip(alpha * img[...,:3].astype(np.float32), dtype, maxval)
        self.image = img

    ### brightness shift
    def brightness_shift(self, alpha):
        img = self.image
        maxval = np.max(img[..., :3])
        dtype = img.dtype
        img[..., :3] = clip(alpha * 255 + img[...,:3].astype(np.float32), dtype, maxval)
        self.image = img

    ### contrast
    def contrast(self, alpha):
        img = self.image
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        maxval = np.max(img[..., :3])
        dtype = img.dtype
        img[:, :, :3] = clip(alpha * img[:, :, :3].astype(np.float32) + gray, dtype, maxval)
        self.image = img      

    ### gamma correction
    def gamma_correction(self, gamma):   
        image = self.image
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 
                for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, table) # apply gamma correction using the lookup table  
        self.image = image

    ### histogram ecualization
    def clahe(self, clipLimit, tileGridSize):
        image = self.image
        img_yuv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        self.image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)



    #geometric transforms



    #pytorch transform
    def to_tensor(self):
        pass



class ObjectImageTransform(ObjectTransform):
    def __init__(self, image, label ):
        """
        Arg:
            @image
            @label
        """
        super(ObjectImageTransform, self).__init__(image)
        self.label = label

    #pytorch transform
    def to_tensor(self):
        image  = self.image
        label  = self.label

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image':  torch.from_numpy(image).float() ,
                'labels': torch.from_numpy(label).float() }


class ObjectImageAndMaskTransform(ObjectTransform):
    def __init__(self, image, mask ):
        """
        Arg:
            @image
            @mask
        """
        super(ObjectImageAndMaskTransform, self).__init__(image)
        self.mask = mask

    
    #pytorch transform
    def to_tensor(self):
        image  = self.image
        mask  = (self.mask>0).astype(np.uint8)

        # numpy image: H x W x C
        # torch image: C X H X W        
        image  = image.transpose((2, 0, 1)).astype(np.float)
        mask   = label.transpose((2, 0, 1)).astype(np.float)

        return {'image':  torch.from_numpy(image).float() ,
                'labels': torch.from_numpy(mask).float() }

    #geometric transformation
    def to_unet_input( self, fov_size=388, padding_mode = cv2.BORDER_CONSTANT ):
        self.image = F.resize_unet_transform(self.image, fov_size, cv2.INTER_LINEAR,  padding_mode)
        self.mask  = F.resize_unet_transform(self.mask , fov_size, cv2.INTER_NEAREST, padding_mode)


        


class ObjectImageMaskAndWeightTransform(ObjectImageAndMaskTransform):
    def __init__(self, image, mask, weight ):
        """
        Arg:
            @image
            @mask
            @weight
        """
        super(ObjectImageMaskAndWeightTransform, self).__init__(image, mask)
        self.weight = weight

    
    #pytorch transform
    def to_tensor(self):
        
        image  = self.image
        mask   = (self.mask>0).astype(np.uint8)
        weight = self.weight

        # numpy image: H x W x C
        # torch image: C X H X W        
        image  = image.transpose((2, 0, 1)).astype(np.float)
        mask  = mask.transpose((2, 0, 1)).astype(np.float)
        weight = weight.transpose((2, 0, 1)).astype(np.float)

        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(mask).float(),
                'weight': torch.from_numpy(weight).float(),
               }

    #geometric transformation
    def to_unet_input( self, fov_size=388, padding_mode = cv2.BORDER_CONSTANT ):
        super(ObjectImageMaskAndWeightTransform, self).to_unet_input(fov_size, padding_mode)
        self.weight = F.resize_unet_transform(self.weight, fov_size, cv2.INTER_LINEAR,  padding_mode)
