
import os
import shutil
import time
import numpy as np
import math
import scipy.misc
from skimage import color

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as nnfun


from . import netmodels as nnmodels
from . import netlosses as nloss
from . import torchutl
from . import graphic as gph
from . import netutility as nutl
from . import netlearningrate

import cv2



class Network(object):
    """
    Convolutional Net
        -training
        -val
        -test
    """

    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel = False,
        seed=1,
        print_freq=10,
        gpu=0
        ):
        """
        Initialization
            -patchproject (str)
            -nameproject (str)
            -no_cuda (bool) (default is True)
            -seed (int)
            -print_freq (int)
            -gpu (int)
        """

        # cuda
        self.cuda = not no_cuda and torch.cuda.is_available()
        self.parallel = not no_cuda and parallel
        torch.manual_seed(seed)

        if self.cuda:
            torch.cuda.set_device(gpu)
            torch.cuda.manual_seed(seed)

        # create project
        self.nameproject = nameproject
        self.pathproject = os.path.join(patchproject, nameproject)
        self.pathmodels = os.path.join(self.pathproject, 'models')
        if not os.path.exists(self.pathproject):
            os.makedirs(self.pathproject)
        if not os.path.exists(self.pathmodels):
            os.makedirs(self.pathmodels)

        # Set the graphic visualization
        self.plotter = gph.VisdomLinePlotter(env_name=nameproject)
        self.visheatmap = gph.HeatMapVisdom(env_name=nameproject)
        self.visimshow = gph.ImageVisdom(env_name=nameproject)

        self.print_freq = print_freq

        self.num_classes = 0
        self.lr = 0.0001
        self.start_epoch = 0
        self.imsize = 0

        self.arch = ''
        self.opt = ''
        self.lrsch = ''

        self.net = None
        self.criterion = None
        self.optimizer = None
        self.lrscheduler = None
        self.vallosses = None

        self.accuracy = nloss.Accuracy()
        self.dice = nloss.Dice()
  
    def create(self, arch, num_classes, loss, lr, momentum, opt, lrsch, pretrained=False ):
        """
        Create            
            -arch (string): architecture
            -loss:
            -lr (float): learning rate
            -opt: 
            -lrsch: scheduler learning rate
        """
        
        self._create_model( arch, num_classes, pretrained )
        self._create_loss( loss )
        self._create_optimizer( opt, lr, momentum )
        self._create_scheduler_lr( lrsch )

    def training(self, data_loader, epoch=0):

        data_time  = torchutl.AverageMeter()
        batch_time = torchutl.AverageMeter()
        losses     = torchutl.AverageMeter()
        accs_t     = torchutl.AverageMeter()
        accs_for   = torchutl.AverageMeter()
        accs_bak   = torchutl.AverageMeter()
        accs_edg   = torchutl.AverageMeter()
        dices      = torchutl.AverageMeter()

        # switch to evaluate mode
        self.net.train()

        end = time.time()
        for i, sample in enumerate(data_loader):
            
            # measure data loading time
            data_time.update(time.time() - end)
            # get data (image, label, weight)
            inputs, targets, weights = sample['image'], sample['label'], sample['weight']
            batch_size = inputs.size(0)

            if self.cuda:
                targets = targets.cuda(async=True)
                inputs_var  = Variable(inputs.cuda(),  requires_grad=False)
                targets_var = Variable(targets.cuda(), requires_grad=False)
                weights_var = Variable(weights.cuda(), requires_grad=False)
            else:
                inputs_var  = Variable(inputs,  requires_grad=False)
                targets_var = Variable(targets, requires_grad=False)
                weights_var = Variable(weights, requires_grad=False)

            # fit (forward)
            outputs = self.net(inputs_var)

            # evaluate criterio
            loss = self.criterion(outputs, targets_var, weights_var)
            
            # measure accuracy and record loss
            acc_t, acc_for, acc_bak, acc_edg = self.accuracy(outputs, targets_var )
            dice = self.dice( outputs, targets_var )

            losses.update(loss.data[0], inputs.size(0))
            accs_t.update(acc_t, inputs.size(0))
            accs_for.update(acc_for, inputs.size(0))
            accs_bak.update(acc_bak, inputs.size(0))
            accs_edg.update(acc_edg, inputs.size(0))
            dices.update(dice.data[0], inputs.size(0))
  
            # optimizer
            self.optimizer.zero_grad()
            (loss*batch_size).backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:  

                strinfo  = '|Train: {:4d}|{:4d}|{:4d} '
                strinfo += '|time: {batch_time.val:8.4f} '     
                strinfo += '|loss: {loss.val:8.4f} '
                strinfo += '|acc: {acc.val:8.4f} ' 
                strinfo += '|dice: {dice.val:8.4f} '
                
                print(
                    strinfo.format(
                        epoch, i, len(data_loader),
                        batch_time=batch_time,
                        loss=losses,
                        acc=accs_t,
                        dice=dices                        
                    ),
                    flush=True                
                    )

                #============ Visdom logging ============#
                # (1) Log the scalar values
                info = {
                    'loss':{'loss':losses}, 
                    'metric': {'acc_t':accs_t, 'acc_for':accs_for, 'acc_bak':accs_bak, 'acc_edg':accs_edg, 'dice':dices }
                    }

                for tag, value in info.items():
                    for k,v in value.items():
                        self.plotter.plot(tag, 'tr_{}'.format(k), epoch + float(i+1)/len(data_loader), v.avg) 


    def evaluate(self, data_loader, epoch=0 ):
        
        batch_time = torchutl.AverageMeter()
        losses     = torchutl.AverageMeter()
        accs_t     = torchutl.AverageMeter()
        accs_for   = torchutl.AverageMeter()
        accs_bak   = torchutl.AverageMeter()
        accs_edg   = torchutl.AverageMeter()
        dices      = torchutl.AverageMeter()

        # switch to evaluate mode
        self.net.eval()

        end = time.time()
        for i, sample in enumerate(data_loader):
            
            # get data (image, label)
            inputs, targets, weights = sample['image'], sample['label'], sample['weight']

            if self.cuda:
                targets = targets.cuda(async=True)
                inputs_var  = Variable(inputs.cuda(),  requires_grad=False, volatile=True)
                targets_var = Variable(targets.cuda(), requires_grad=False, volatile=True)
                weights_var = Variable(weights.cuda(), requires_grad=False, volatile=True)
            else:
                inputs_var  = Variable(inputs,  requires_grad=False, volatile=True)
                targets_var = Variable(targets, requires_grad=False, volatile=True)
                weights_var = Variable(weights, requires_grad=False, volatile=True)
            
            # fit (forward)
            outputs = self.net(inputs_var)

            # evaluate criterio
            loss = self.criterion(outputs, targets_var, weights_var)

            # measure accuracy and record loss
            acc_t, acc_for, acc_bak, acc_edg = self.accuracy(outputs, targets_var )
            dice = self.dice( outputs, targets_var )

            losses.update(loss.data[0], inputs.size(0))
            accs_t.update(acc_t, inputs.size(0))
            accs_for.update(acc_for, inputs.size(0))
            accs_bak.update(acc_bak, inputs.size(0))
            accs_edg.update(acc_edg, inputs.size(0))
            dices.update(dice.data[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                strinfo  = '|Valdt: {:4d}|{:4d}|{:4d} '                
                strinfo += '|time: {batch_time.val:8.4f} '     
                strinfo += '|loss: {loss.val:8.4f} '
                strinfo += '|acc: {acc.val:8.4f} '  
                strinfo += '|dice: {dice.val:8.4f} '

                print(
                    strinfo.format(
                        epoch, i, len(data_loader),
                        batch_time=batch_time,
                        loss=losses,
                        acc=accs_t,
                        dice=dices                      
                        ) ,
                    flush=True               
                    )

        #save validation loss
        self.vallosses = losses

        #============ Visdom logging ============#
        # (1) Log the scalar values
        info = {
            'loss':{'loss':losses}, 
            'metric': {'acc_t':accs_t, 'acc_for':accs_for, 'acc_bak':accs_bak, 'acc_edg':accs_edg, 'dice':dices    }
            }

        for tag, value in info.items():
            for k,v in value.items():
                self.plotter.plot(tag, 'ts_{}'.format(k), epoch, v.avg) 

        # print(' * Prec@1 {top1.avg:.3f} Prec@2 {top2.avg:.3f}'
        #         .format(top1=top1, top2=top2))

        strinfo  = '|Valdt: {:4d}|{:4d}|{:4d} '                
        strinfo += '|time: {batch_time.avg:8.4f} '     
        strinfo += '|loss: {loss.avg:8.4f} '
        strinfo += '|acc: {acc.avg:8.4f} '
        strinfo += '|dice: {dice.avg:8.4f} '

        print(
            strinfo.format(
                epoch, i, len(data_loader),
                batch_time=batch_time,
                loss=losses,
                acc=accs_t,
                dice=dices                   
                ) ,
            flush=True               
            )
        
        #vizual_freq
        if epoch % 10 == 0:

            ws,hw = 100,100
            prob = nnfun.softmax(outputs,dim=1)
            prob = prob.data[0]
            _,maxprob = torch.max(prob,0)
            
            self.visheatmap.show('Label', targets_var.data.cpu()[0].numpy()[1,:,:] )
            self.visheatmap.show('Weight map', weights_var.data.cpu()[0].numpy()[0,:,:])
            self.visheatmap.show('Image', inputs_var.data.cpu()[0].numpy()[0,:,:])
            self.visheatmap.show('Max prob',maxprob.cpu().numpy().astype(np.float32) )
            for k in range(prob.shapeweight[0]):                
                self.visheatmap.show('Heat map {}'.format(k), prob.cpu()[k].numpy() )
           

        return accs_t.avg


    def test(self, data_loader):
        pass


    def inference(self, image):  
        pass      


    def representation(self, data_loader):
        pass


    
    def _create_model(self, arch='simplenet', num_classes=3, pretrained=False):
        """
        Create model
            -arch: select architecture
            -num_classes
            -pretrained

        """        
        self.net = None       

        # Pytorch Archtectures 
        #--------------------------------------------------------------------------------------------
        if arch == 'unet':
            self.net = nnmodels.unet( n_classes = num_classes )  
        elif arch == 'unet11':
            self.net = nnmodels.unet11( num_classes = num_classes ) 
        elif arch == 'dunet':
            self.net = nnmodels.dunet( n_classes = num_classes )                   
        else:
            assert(False)
        
        self.arch = arch
        self.num_classes = num_classes
        if self.cuda == True:
            self.net.cuda()

        if self.parallel == True and self.cuda == True:
            self.net = nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
    

    def _create_loss(self, loss='wmcedice'):
        # create loss

        if loss == 'wmce':
            self.criterion = nloss.WeightedMCEloss()
        elif loss == 'bdice':
            self.criterion = nloss.BDiceLoss()
        elif loss == 'wbdice':
            self.criterion = nloss.WeightedBDiceLoss()
        elif loss == 'wmcedice':
            self.criterion = nloss.WeightedMCEDiceLoss()
        elif loss == 'wfocalmce':
            self.criterion = nloss.WeightedMCEFocalloss()
        elif loss == 'mcedice':
            self.criterion = nloss.MCEDiceLoss()            
        else:
            assert(False)
        
        

    def _create_optimizer(self, opt='adam', lr=0.0001, momentum=0.99):
        
        # create optimizer
        if opt == 'adam':
            self.optimizer = torch.optim.Adam( self.net.parameters(), lr=lr)
        elif opt == 'sgd':
            self.optimizer = torch.optim.SGD( self.net.parameters(), lr=lr, momentum=momentum)
        elif opt == 'rprop':
            self.optimizer = torch.optim.Rprop( self.net.parameters(), lr=lr) 
        elif opt == 'rmsprop':
            self.optimizer = torch.optim.RMSprop( self.net.parameters(), lr=lr)           
        else:
            assert(False)

        self.lr = lr; 
        self.momentum = momentum
        self.opt=opt

    def _create_scheduler_lr(self, lrsch ):
        
        #MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        #ExponentialLR
        #CosineAnnealingLR

        self.lrscheduler = None

        if lrsch == 'fixed':
            pass           
        elif lrsch == 'step':
            self.lrscheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1 )
        elif lrsch == 'cyclic': 
            self.lrscheduler = netlearningrate.CyclicLR(self.optimizer)
        elif lrsch == 'exp':
            self.lrscheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99 )
        elif lrsch == 'plateau':
            self.lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=100)
        else:
            assert(False)
        
        self.lrsch = lrsch

    def adjust_learning_rate(self, epoch):
        """
        Update learning rate
        """        
        
        # update
        if self.lrsch == 'fixed':
            lr = self.lr
        elif self.lrsch == 'plateau':
            self.lrscheduler.step( self.vallosses.val )
            for param_group in self.optimizer.param_groups:
                lr = float(param_group['lr'])
                break
            #return
        else:                    
            self.lrscheduler.step() 
            lr = self.lrscheduler.get_lr()[0]        

        # draw
        self.plotter.plot('lr', 'learning rate', epoch, lr )

    def resume(self, pathnammodel):
        """
        Resume: optionally resume from a checkpoint
        """ 
        net = self.net.module if self.parallel else self.net
        start_epoch, prec = torchutl.resumecheckpoint( 
            pathnammodel, 
            net, 
            self.optimizer 
            )
        self.start_epoch = start_epoch
        return start_epoch, prec

    def save(self, epoch, prec, is_best=False, filename='checkpoint.pth.tar'):
        """
        Save model
        """
        net = self.net.module if self.parallel else self.net
        torchutl.save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': self.arch,
                'num_classes': self.num_classes,
                'state_dict': net.state_dict(),
                'prec': prec,
                'optimizer' : self.optimizer.state_dict(),
            }, 
            is_best,
            self.pathmodels,
            filename
            )
    
    def load(self, pathnamemodel):
        bload = False
        if pathnamemodel:
            if os.path.isfile(pathnamemodel):
                print("=> loading checkpoint '{}'".format(pathnamemodel))
                
                if self.cuda:               
                    checkpoint = torch.load( pathnamemodel )
                else:
                    checkpoint = torch.load( pathnamemodel, map_location=lambda storage, loc: storage )

                self._create_model(checkpoint['arch'], checkpoint['num_classes'])
                self.net.load_state_dict( checkpoint['state_dict'] )
                
                print("=> loaded checkpoint for {} arch!".format(checkpoint['arch']))
                bload = True

            else:
                print("=> no checkpoint found at '{}'".format(pathnamemodel))
        
        return bload
   

    def __str__(self): 
        return str(
                'Name: {} \n'
                'arq: {} \n'
                'lr: {} \n'
                'Model: \n{} \n'.format(
                self.nameproject,
                self.arch,
                self.lr,
                self.net
                )
                )


    
        

    
