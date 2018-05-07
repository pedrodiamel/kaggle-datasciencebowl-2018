
import numpy as np
import torch

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

    #colors transforms

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
        mask  = label.transpose((2, 0, 1)).astype(np.float)

        return {'image':  torch.from_numpy(image).float() ,
                'labels': torch.from_numpy(mask).float() }


class ObjectImageMaskAndWeightTransform(ObjectTransform):
    def __init__(self, image, mask, weight ):
        """
        Arg:
            @image
            @mask
            @weight
        """
        super(ObjectImageMaskAndWeightTransform, self).__init__(image)
        self.mask = mask
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

