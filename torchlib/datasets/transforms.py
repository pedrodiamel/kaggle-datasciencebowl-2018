
import random
import numpy as np
import cv2

from .renderblur import BlurRender
from .aumentation import ObjectTransform
from . import functional as F


class ToTransform(object):
    """Generic transform 
    """
    
    def __init__(self, prob):
        """Initialization
        Args:
            @prob: probability
        """
        self.prob=prob
        
        
    def __call__(self,obj):
        if random.random() < self.prob:
            obj = self._execute( obj )
        return obj
    
    def _execute(self,obj):
        pass
    
    def __str__(self):
        return self.__class__.__name__


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, obj):
        return obj.to_tensor()


class ToLinealMotionBlur(ToTransform):
    """Lineal Blur randomly.
    """

    def __init__(self, lmax=100, prob=0.5 ):        
        """Initialization
        Args:
            @lmax: maximun lineal blur
            @prob: probability
        """
        super(ToLinealMotionBlur,self).__init__(prob)
        self.gen = BlurRender(lmax)


    def _execute(self, obj):
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
        texp=0.75,     
        prob=0.5 ):        
        """Initialization
        Args:
            @pSFsize: kernel size (psf)
            @maxTotalLength: length trayectory
            @anxiety:
            @numT:
            @texp:
            @prob: probability
        """
        super(ToMotionBlur,self).__init__(prob)
        self.gen = BlurRender(pSFsize, maxTotalLength, anxiety, numT, texp)


    def _execute(self, obj):
        obj.motion_blur(self.gen)
        return obj



class ToGaussianBlur(ToTransform):
    """Gaussian Blur randomly.
    """

    def __init__(self, prob=0.5, sigma=0.2 ):        
        """Initialization
        Args:
            @lmax: maximun lineal blur
            @prob: probability
        """
        super(ToGaussianBlur,self).__init__(prob)
        self.sigma = sigma

    def _execute(self, obj):
        
        # add gaussian noise
        H,W = obj.shape[:2]
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
    def __init__(self, prob=0.5, limit=0.1 ):        
        """Initialization
        Args:
            @limit: limit
            @prob: probability
        """
        super(RandomBrightness,self).__init__(prob)
        self.limit = limit

    def _execute(self, obj):
        alpha = 1.0 + self.limit*random.uniform(-1, 1)
        obj.brightness(alpha)
        return obj

class RandomBrightnessShift(ToTransform):
    """Random Brightness Shift.
    """
    def __init__(self, prob=0.5, limit=0.01 ):        
        """Initialization
        Args:
            @limit: limit
            @prob: probability
        """
        super(RandomBrightnessShift,self).__init__(prob)
        self.limit = limit

    def _execute(self, obj):
        alpha = 1.0 + self.limit*random.uniform(-1, 1)
        obj.brightness_shift(alpha)
        return obj

class RandomContrast(ToTransform):
    """Random Contrast.
    """
    def __init__(self, prob=0.5, limit=0.1 ):        
        """Initialization
        Args:
            @limit: limit
            @prob: probability
        """
        super(RandomContrast,self).__init__(prob)
        self.limit = limit

    def _execute(self, obj):
        alpha = 1.0 + self.limit*random.uniform(-1, 1)
        obj.brightness_shift(alpha)
        return obj

class RandomGamma(ToTransform):
    """Random Gamma.
    """
    def __init__(self, prob=0.5, limit=0.1 ):        
        """Initialization
        Args:
            @limit: limit
            @prob: probability
        """
        super(RandomGamma,self).__init__(prob)
        self.limit = limit

    def _execute(self, obj):
        alpha = 1.0 + self.limit*random.uniform(-1, 1)
        obj.brightness_shift(alpha)
        return obj


class CLAHE(ToTransform):
    """Random Gamma.
    """
    def __init__(self, prob=0.5, clipLimit=2.0, tileGridSize=(8, 8) ):        
        """Initialization
        Args:
            @limit: limit
            @prob: probability
        """
        super(CLAHE,self).__init__(prob)
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def _execute(self, obj):
        obj.clahe(alpha)
        return obj



# geometrical transforms

class ToResizeUNetFoV(object):
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
    
    def __str__(self):
        return self.__class__.__name__



class CenterCrop(object):
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
    
    def __str__(self):
        return self.__class__.__name__


class RandomCrop(object):
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
    
    def __str__(self):
        return self.__class__.__name__


class RandomScale(ToTransform):
    """ Random Scale.
    """
    def __init__(self, prob=0.5, sxy=0.1, 
        padding_mode=cv2.BORDER_CONSTANT,  
        ):        
        """Initialization
        Args:
            @prob: probability
            @sxy: scale
            @padding_mode        
        """
        super(RandomScale,self).__init__(prob)
        self.sxy = sxy
        self.padding_mode = padding_mode

    def _execute(self, obj):
        
        height, width = obj.size()[:2]
        sxy =  1.0 + self.sxy*random.uniform(-1.0, 1.0)
        obj.scale( sxy, self.padding_mode )
        return obj