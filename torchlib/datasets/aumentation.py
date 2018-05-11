
import torch
import numpy as np
import cv2

from . import utility as utl
from . import functional as F

class ObjectTransform(object):
    def __init__(self, image ):
        self.image = image

    def size(self): return self.image.shape

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

    def crop( self, box):
        """Crop: return if validate crop
        """
        self.image = F.imcrop( self.image, box )
        return True

    def scale( self, sxy, padding_mode = cv2.BORDER_CONSTANT ):
        self.image = F.scale( self.image, sxy, cv2.INTER_LINEAR, padding_mode )

    def hflip(self):
        self.image = F.hflip( self.image )

    def vflip(self):
        self.image = F.vflip( self.image )

    def rotate90(self):
        self.image = F.rotate90( self.image )

    def rotate180(self):
        self.image = F.rotate180( self.image )

    def rotate279(self):
        self.image = F.rotate270( self.image )


    def applay_geometrical_transform(self, mat_r, mat_t, mat_w ):        
        self.image = F.applay_geometrical_transform( self.image, mat_r, mat_t, mat_w, cv2.INTER_LINEAR )
        return True

    def applay_elastic_transform(self, mapx, mapy):        
        self.image  = self._draw_grid( self.image, grid_size=50 )
        self.image  = cv2.remap(self.image,  mapx, mapy, cv2.INTER_CUBIC)
        
        
    # resize unet input
    def to_unet_input( self, fov_size=388, padding_mode = cv2.BORDER_CONSTANT ):
        self.image = F.resize_unet_transform(self.image, fov_size, cv2.INTER_LINEAR,  padding_mode)


    #pytorch transform
    def to_tensor(self):
        pass


    # Aux function to draw a grid
    def _draw_grid(self, imgrid, grid_size=50, color=(255,0,0), thickness=1):
        
        m,n = imgrid.shape[:2]

        # Draw grid lines
        for i in range(0, n-1, grid_size):
            cv2.line(imgrid, (i+grid_size, 0), (i+grid_size, m), color=color, thickness=thickness)
        for j in range(0, m-1, grid_size):
            cv2.line(imgrid, (0, j+grid_size), (n, j+grid_size), color=color, thickness=thickness)

        return imgrid



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
