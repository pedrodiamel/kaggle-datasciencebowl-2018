import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def to_one_hot(mask, size):
    
    n, c, h, w = size
    ymask = torch.FloatTensor(size).zero_()
    new_mask = torch.LongTensor(n,1,h,w)
    if mask.is_cuda:
        ymask = ymask.cuda(mask.get_device())
        new_mask = new_mask.cuda(target.get_device())
    new_mask[:,0,:,:] = torch.clamp(mask.data, 0, c-1)
    ymask.scatter_(1, new_mask , 1.0)    
    return Variable(ymask)

def centercrop(image, w, h):        
    nt, ct, ht, wt = image.size()
    padw, padh = (wt-w) // 2 ,(ht-h) // 2
    if padw>0 and padh>0: image = image[:,:, padh:-padh, padw:-padw]
    return image

def flatten(x):        
    x_flat = x.clone()
    x_flat = x_flat.view(x.shape[0], -1)
    return x_flat
    
class WeightedMCEloss(nn.Module):

    def __init__(self ):
        super(WeightedMCEloss, self).__init__()


    def forward(self, y_pred, y_true, weight ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h )
        weight = centercrop(weight, w, h )
        
        y_pred_log =  F.log_softmax(y_pred, dim=1)
        logpy = torch.sum( weight * y_pred_log * y_true, dim=1 )
        #loss  = -torch.sum(logpy) / torch.sum(weight)
        loss  = -torch.mean(logpy)

        return loss

class WeightedMCEFocalloss(nn.Module):
    
    def __init__(self, gamma=2.0 ):
        super(WeightedMCEFocalloss, self).__init__()
        self.gamma = gamma

    def forward(self, y_pred, y_true, weight ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h )
        weight = centercrop(weight, w, h )
        
        y_pred_log =  F.log_softmax(y_pred, dim=1)

        fweight = (1 - F.softmax(y_pred, dim=1) ) ** self.gamma
        weight  = weight*fweight

        logpy = torch.sum( weight * y_pred_log * y_true, dim=1 )
        #loss  = -torch.sum(logpy) / torch.sum(weight)
        loss  = -torch.mean(logpy)
        
        return loss

class WeightedBCELoss(nn.Module):
    
    def __init__(self ):
        super(WeightedBCELoss, self).__init__()

    def forward(self, y_pred, y_true, weight ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        weight = centercrop(weight, w, h )
     
        logit_y_pred = torch.log(y_pred / (1. - y_pred))
        loss = weight * (logit_y_pred * (1. - y_true) + 
                        torch.log(1. + torch.exp(-torch.abs(logit_y_pred))) + torch.clamp(-logit_y_pred, min=0.))
        loss = torch.sum(loss) / torch.sum(weight)

        return loss

class BCELoss(nn.Module):
    
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true ):        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        loss = self.bce(y_pred, y_true)
        return loss

class WeightedBDiceLoss(nn.Module):
    
    def __init__(self ):
        super(WeightedBDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()


    def forward(self, y_pred, y_true, weight ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        weight = centercrop(weight, w, h )

        y_pred = self.sigmoid(y_pred)

        smooth = 1.
        w, m1, m2 = weight, y_true, y_pred
        score = (2. * torch.sum(w * m1 * m2) + smooth) / (torch.sum(w * m1) + torch.sum(w * m2) + smooth)
        loss = 1. - torch.sum(score)

        return loss


class BDiceLoss(nn.Module):
    
    def __init__(self):
        super(BDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true, weight=None ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)

        y_pred = self.sigmoid(y_pred)

        smooth = 1.
        y_true_f = flatten(y_true)
        y_pred_f = flatten(y_pred)
        score = (2. * torch.sum(y_true_f * y_pred_f) + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return 1. - score


class BLogDiceLoss(nn.Module):
    
    def __init__(self, classe = 1 ):
        super(BLogDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classe = classe

    def forward(self, y_pred, y_true, weight=None ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        y_pred = self.sigmoid(y_pred)

        eps = 1e-15
        dice_target = (y_true[:,self.classe,...] == 1).float()
        dice_output = y_pred[:,self.classe,...]
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum() + eps

        return -torch.log(2 * intersection / union)

class WeightedMCEDiceLoss(nn.Module):
    
    def __init__(self, alpha=1.0, gamma=1.0  ):
        super(WeightedMCEDiceLoss, self).__init__()
        self.loss_mce = WeightedMCEFocalloss()
        self.loss_dice = BLogDiceLoss()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, weight ):        
        
        alpha = self.alpha
        weight = torch.pow(weight,self.gamma)
        loss_dice = self.loss_dice(y_pred, y_true)
        loss_mce = self.loss_mce(y_pred, y_true, weight)
        loss = loss_mce + alpha*loss_dice        
        return loss

class MCEDiceLoss(nn.Module):
    
    def __init__(self, alpha=1.0, gamma=1.0  ):
        super(MCEDiceLoss, self).__init__()
        self.loss_mce = BCELoss()
        self.loss_dice = BLogDiceLoss()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, weight ):        
        
        alpha = self.alpha                
        loss_dice = self.loss_dice(y_pred, y_true)        
        loss_fg  = self.loss_mce( y_pred[:,1,...].unsqueeze(1), y_true[:,1,...].unsqueeze(1) )
        loss_th  = self.loss_mce( y_pred[:,2,...].unsqueeze(1), y_true[:,2,...].unsqueeze(1) )
        loss = 0.5*loss_fg + 0.5*loss_th + alpha*loss_dice     
        return loss



class Accuracy(nn.Module):
    
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, input, target ):
        
        n, ch, h, w = input.size()
        nt,ct, ht,wt= target.size()
        
        target = centercrop(target, w, h)

        prob = F.softmax(input, dim=1)
        prob = prob.data
        _, maxprob = torch.max(prob,1)
                
        correct = torch.zeros(ch)
        for c in range(ch):
            ctarget = target[:,c,...]
            num = (((maxprob.eq(c) + ctarget.data.eq(1)).eq(2)).float().sum() + 1 )
            den = (ctarget.data.eq(1).float().sum() + 1)
            correct[c] = num/den

        forg_acc  =  100.0*correct[1]
        back_acc  =  100.0*correct[0]
        edge_acc  =  100.0*correct[2]
        total_acc =  ( forg_acc + back_acc + edge_acc )/ch

        return total_acc, forg_acc, back_acc, edge_acc


class Dice(nn.Module):
    
    def __init__(self):
        super(Dice, self).__init__()
        self.sigmoid = nn.Sigmoid()


    def forward(self, y_pred, y_true ):
        
        n, ch, h, w = y_pred.size()
        y_true   = centercrop(y_true, w, h)

        prob = F.softmax(y_pred, dim=1)
        prob = prob.data
        _, maxprob = torch.max(prob,1)

        y_true_f = flatten(y_true[:,1,...]).float()
        y_pred_f = flatten(maxprob).float()
        
        # psilon = 1e-15
        # intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
        # union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1) + epsilon
        # return 2 * (intersection / union).mean()
        
        #y_pred = self.sigmoid(y_pred)        
        # y_true_f = flatten(y_true)
        # y_pred   = y_pred.float()
        # y_pred_f = torch.gt(flatten(y_pred), 0.5).float()
        
        intersection = y_true_f * y_pred_f
        score = 2. * torch.sum(intersection) / (torch.sum(y_true_f) + torch.sum(y_pred_f))

        return 100*score

