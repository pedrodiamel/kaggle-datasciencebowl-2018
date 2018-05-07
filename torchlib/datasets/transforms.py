
import random

from .renderblur import BlurRender
from .aumentation import ObjectTransform


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


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, obj):
        return obj.to_tensor()
