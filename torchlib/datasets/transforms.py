
import random
import numpy as np
import cv2

from .renderblur import BlurRender
from .aumentation import ObjectTransform
from . import functional as F



class ToTransform(object):
    """Abstrat class of Generic transform 
    """
    
    def __init__(self):
        pass        
    
    def __str__(self):
        return self.__class__.__name__



class ToRandomTransform(ToTransform):
    """Random transform: 
    """
    
    def __init__(self, tran, prob):
        """Initialization
        Args:
            @tran: class tranform 
            @prob: probability
        """
        self.tran = tran 
        self.prob=prob
        
        
    def __call__(self,obj):
        if random.random() < self.prob:
            obj = self.tran( obj )
        return obj
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, obj):
        return obj.to_tensor()


class ToLinealMotionBlur(ToTransform):
    """Lineal Blur randomly.
    """

    def __init__(self, lmax=100  ):        
        """Initialization
        Args:
            @lmax: maximun lineal blur
        """
        self.gen = BlurRender(lmax)

    def __call__(self, obj):
        obj.lineal_blur(self.gen)
        return obj


class ToMotionBlur(ToTransform):
    """Motion Blur randomly.
    """

    def __init__(self,
        pSFsize=64,
        maxTotalLength=64,
        anxiety=0.005,
        numT=2000,
        texp=0.75, ):        
        """Initialization
        Args:
            @pSFsize: kernel size (psf)
            @maxTotalLength: length trayectory
            @anxiety:
            @numT:
            @texp:
        """
        self.gen = BlurRender(pSFsize, maxTotalLength, anxiety, numT, texp)

    def __call__(self, obj):
        obj.motion_blur(self.gen)
        return obj



class ToGaussianBlur(ToTransform):
    """Gaussian Blur randomly.
    """

    def __init__(self, sigma=0.2 ):        
        """Initialization
        Args:
            @lmax: maximun lineal blur
        """
        self.sigma = sigma

    def __call__(self, obj):
        
        # add gaussian noise
        H,W = obj.size()[:2]
        noise = np.array([random.gauss(0,self.sigma) for i in range(H*W)])
        noise = noise.reshape(H,W)
        obj.add_noise( noise )
        wnd = random.randint(1,3) * 2 + 1
        obj.gaussian_blur(wnd)
        return obj


# Color tranformations

class RandomBrightness(ToTransform):
    """Random Brightness.
    """
    def __init__(self, factor=0.1 ):        
        """Initialization
        Args:
            @factor: factor
        """
        self.factor = factor

    def __call__(self, obj):
        alpha = 1.0 + self.factor*random.uniform(-1, 1)
        obj.brightness(alpha)
        return obj


class RandomBrightnessShift(ToTransform):
    """Random Brightness Shift.
    """
    def __init__(self, factor=0.01 ):        
        """Initialization
        Args:
            @factor: factor
        """
        self.factor = factor

    def __call__(self, obj):
        alpha = 1.0 + self.factor*random.uniform(-1, 1)
        obj.brightness_shift(alpha)
        return obj

class RandomContrast(ToTransform):
    """Random Contrast.
    """
    def __init__(self, factor=0.1 ):        
        """Initialization
        Args:
            @factor: factor
        """
        super(RandomContrast,self).__init__(prob)
        self.limit = limit

    def __call__(self, obj):
        alpha = 1.0 + self.factor*random.uniform(-1, 1)
        obj.brightness_shift(alpha)
        return obj

class RandomGamma(ToTransform):
    """Random Gamma.
    """
    def __init__(self, factor=0.5, limit=0.1 ):        
        """Initialization
        Args:
            @factor: factor
        """
        self.factor = factor

    def __call__(self, obj):
        alpha = 1.0 + self.factor*random.uniform(-1, 1)
        obj.brightness_shift(alpha)
        return obj


class CLAHE(ToTransform):
    """CLAHE ecualization.
    """
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8) ):        
        """Initialization
        Args:
            @factor: factor
        """
        self.clipfactor = clipfactor
        self.tileGridSize = tileGridSize

    def __call__(self, obj):
        obj.clahe( self.clipfactor,  self.tileGridSize )
        return obj



# geometrical transforms

class ToResizeUNetFoV(ToTransform):
    """Resize to unet fov
    """
    
    def __init__(self, fov=388, padding_mode=cv2.BORDER_CONSTANT):
        """Initialization
        Args:
            @fov: size input layer for unet model
        """
        self.fov=fov
        self.padding_mode = padding_mode
        
    def __call__(self,obj):
        obj.to_unet_input( self.fov, self.padding_mode )
        return obj
    


class CenterCrop(ToTransform):
    """Center Crop
    """
    
    def __init__(self, cropsize ):
        """Initialization
        Args:
            @cropsize [w,h]
        """
        self.cropsize = cropsize
        
    def __call__(self, obj):
        h, w = obj.size()[:2]
        x = (w - self.cropsize[0]) // 2
        y = (h - self.cropsize[1]) // 2
        obj.crop( [ x, y, self.cropsize[0], self.cropsize[1] ] )
        return obj
    

class RandomCrop(ToTransform):
    """Random Crop
    """
    
    def __init__(self, cropsize, limit=10 ):
        """Initialization
        Args:
            @cropsize [w,h]
            @limit
        """
        self.cropsize = cropsize
        self.limit = limit
        self.centecrop = CenterCrop(cropsize)
        
    def __call__(self, obj):
        h, w = obj.size()[:2]
        newW,newH = self.cropsize

        assert(w - newW + self.limit > 0)
        assert(h - newH + self.limit > 0)

        for _ in range(10):       
            x = random.randint( -self.limit, (w - newW) + self.limit )
            y = random.randint( -self.limit, (h - newH) + self.limit )
            if obj.crop( [ x, y, self.cropsize[0], self.cropsize[1] ] ):
                return obj

        return self.centecrop(obj)


class RandomScale(ToTransform):
    """ SRandom Scale.
    """
    def __init__(self, factor=0.1, padding_mode=cv2.BORDER_CONSTANT,  
        ):        
        """Initialization
        Args:
            @factor: factor of scale
            @padding_mode        
        """
        self.factor = factor
        self.padding_mode = padding_mode

    def __call__(self, obj):        
        height, width = obj.size()[:2]
        factor =  1.0 + self.factor*random.uniform(-1.0, 1.0)
        obj.scale( factor, self.padding_mode )
        return obj


class HFlip(ToTransform):
    """ Horizontal Flip.
    """
    def __init__(self):        
        """Initialization 
        """
        pass

    def __call__(self, obj):
        obj.hflip()
        return obj

class VFlip(ToTransform):
    """ Vertical Flip.
    """
    def __init__(self):        
        """Initialization 
        """
        pass

    def __call_(self, obj):
        obj.vflip()
        return obj

class Rotate90(ToTransform):
    """ Rotate 90.
    """
    def __init__(self):        
        """Initialization 
        """
        pass

    def __call_(self, obj):
        obj.Rotate90()
        return obj
    
class Rotate180(ToTransform):
    """ Rotate 180 .
    """
    def __init__(self):        
        """Initialization 
        """
        pass

    def __call_(self, obj):
        obj.Rotate180()
        return obj

class Rotate270(ToTransform):
    """ Rotate 270 .
    """
    def __init__(self):        
        """Initialization 
        """
        pass

    def __call_(self, obj):
        obj.Rotate270()
        return obj

class RandomGeometricalTranform(ToTransform):
    """ Random Geometrical Tranform
    """

    def __init__(self, angle=360, translation=0.2, warp=0.0 ):        
        """Initialization 
        Args:
            @angle: ratate angle
            @translate 
            @warp
        """
        self.angle = angle
        self.translation = translation
        self.warp = warp

    def __call__(self, obj):
        
        imsize = obj.size()[:2]
        for _ in range(10):       
            mat_r, mat_t, mat_w = F.get_geometric_random_transform( imsize, self.angle, self.translation, self.warp )
            if obj.applay_geometrical_transform( mat_r, mat_t, mat_w ):
                return obj
        return obj



class RandomElasticDistort(ToTransform):
    """ Random Elastic Distort
    """

    def __init__(self, size_grid=50, deform=15  ):        
        """Initialization 
        Args:
            @size_grid: ratate angle
            @deform 
        """
        self.size_grid = size_grid
        self.deform = deform

    def __call__(self, obj):
        
        imsize = obj.size()[:2]
        mapx, mapy = F.get_elastic_transform(imsize, self.size_grid, self.deform )
        obj.applay_elastic_transform( mapx, mapy )

        return obj
        
