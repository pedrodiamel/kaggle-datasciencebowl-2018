

import os
import math
import shutil
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
import scipy.misc
from tqdm import tqdm

from . import netmodels as nnmodels
from . import netlearningrate
from . import netlosses as nloss
from . import graphic as gph
from . import torchutls as nutl

from .logger import Logger, AverageFilterMeter, AverageMeter


#----------------------------------------------------------------------------------------------
# Abstract Neural Net 

class AbstractNeuralNet(object):
    """
    Abstract Convolutional Neural Net 
    """

    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0
        ):
        """
        Initialization
            -patchproject (str): path project
            -nameproject (str):  name project
            -no_cuda (bool): system cuda (default is True)
            -parallel (bool)
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

        self.print_freq = print_freq
        self.num_input_channels = 0
        self.num_output_channels = 0
        self.size_input = 0
        self.lr = 0.0001
        self.start_epoch = 0        

        self.s_arch = ''
        self.s_optimizer = ''
        self.s_lerning_rate_sch = ''
        self.s_loss = ''

        self.net = None
        self.criterion = None
        self.optimizer = None
        self.lrscheduler = None
        self.vallosses = None

    def create(self, 
        arch, 
        num_output_channels, 
        num_input_channels, 
        loss, 
        lr, 
        momentum, 
        optimizer, 
        lrsch, 
        pretrained=False
        ):
        """
        Create            
            -arch (string): architecture
            -loss (string):
            -lr (float): learning rate
            -optimizer (string) : 
            -lrsch (string): scheduler learning rate
            -pretrained (bool)
        """
                
        self.s_arch = arch
        self.s_optimizer = optimizer
        self.s_lerning_rate_sch = lrsch
        self.s_loss = loss

        self._create_model( arch, num_output_channels, num_input_channels, pretrained )
        self._create_loss( loss )
        self._create_optimizer( optimizer, lr, momentum )
        self._create_scheduler_lr( lrsch )

    def training(self, data_loader, epoch=0):
        pass

    def evaluate(self, data_loader, epoch=0):
        pass

    def test(self, data_loader):
        pass

    def inference(self, image):        
        pass

    def representation(self, data_loader):
        pass
    
    def fit( self, train_loader, val_loader, epochs=100, snapshot=10 ):

        best_prec = 0
        print('\nEpoch: {}/{}(0%)'.format(self.start_epoch, epochs))
        print('-' * 25)

        self.evaluate(val_loader, epoch=self.start_epoch)        
        for epoch in range(self.start_epoch, epochs):       

            try:
                
                self._to_beging_epoch(epoch, epochs, train_loader, val_loader)

                self.adjust_learning_rate(epoch)     
                self.training(train_loader, epoch)

                print('\nEpoch: {}/{} ({}%)'.format(epoch,epochs, int((float(epoch)/epochs)*100) ) )
                print('-' * 25)
                
                prec = self.evaluate(val_loader, epoch+1 )            

                # remember best prec@1 and save checkpoint
                is_best = prec > best_prec
                best_prec = max(prec, best_prec)
                if epoch % snapshot == 0 or is_best or epoch==(epochs-1) :
                    self.save(epoch, best_prec, is_best, 'chk{:06d}.pth.tar'.format(epoch))

                self._to_end_epoch(epoch, epochs, train_loader, val_loader)

            except KeyboardInterrupt:
                
                print('Ctrl+C, saving snapshot')
                is_best = False
                best_prec = 0
                self.save(epoch, best_prec, is_best, 'chk{:06d}.pth.tar'.format(epoch))
                return

    def _to_beging_epoch(self, epoch, epochs, train_loader, val_loader):
        pass

    def _to_end_epoch(self, epoch, epochs, train_loader, val_loader):
        pass


    def _create_model(self, arch, num_output_channels, num_input_channels, pretrained):
        """
        Create model
            -arch (string): select architecture
            -num_classes (int)
            -pretrained (bool)

        """    
        pass

    def _create_loss(self, loss):
        """
        Create loss
            -loss (string): select loss function
        """
        pass

    def _create_optimizer(self, optimizer='adam', lr=0.0001, momentum=0.99):
        """
        Create optimizer
            -optimizer (string): select optimizer function
            -lr (float): learning rate
            -momentum (float): momentum
        """
        
        self.optimizer = None

        # create optimizer
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam( self.net.parameters(), lr=lr)
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD( self.net.parameters(), lr=lr, momentum=momentum)
        elif optimizer == 'rprop':
            self.optimizer = torch.optim.Rprop( self.net.parameters(), lr=lr) 
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop( self.net.parameters(), lr=lr)           
        else:
            assert(False)

        self.lr = lr; 
        self.momentum = momentum
        self.s_optimizer = optimizer

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
            self.lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10)
        else:
            assert(False)
        
        self.s_lerning_rate_sch = lrsch

    def adjust_learning_rate(self, epoch):
        """
        Update learning rate
        """       
 
        # update
        if self.s_lerning_rate_sch == 'fixed': lr = self.lr
        elif self.s_lerning_rate_sch == 'plateau':
            self.lrscheduler.step( self.vallosses.val )
            for param_group in self.optimizer.param_groups:
                lr = float(param_group['lr'])
                break            
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
        start_epoch, prec = nutl.resumecheckpoint( 
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
        print('>> save model epoch {} ({}) in {}'.format(epoch, prec, filename))
        net = self.net.module if self.parallel else self.net
        nutl.save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': self.s_arch,
                'imsize': self.size_input,
                'num_classes': self.num_output_channels,
                'num_channels': self.num_input_channels,
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
                checkpoint = torch.load( pathnamemodel ) if self.cuda else torch.load( pathnamemodel, map_location=lambda storage, loc: storage )
                
                self._create_model(checkpoint['arch'], checkpoint['num_classes'], checkpoint['num_channels'], False )                
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
                'loss: {} \n'
                'optimizer: {} \n'
                'lr: {} \n'
                'size input: {} \n'
                'num input channels {} \n'
                'num output channels: {} \n'
                'Model: \n{} \n'.format(
                    self.nameproject,
                    self.s_arch,
                    self.s_loss,
                    self.s_optimizer,
                    self.lr,
                    self.size_input,
                    self.num_input_channels,
                    self.num_output_channels,
                    self.net
                    )
                )


#----------------------------------------------------------------------------------------------
# Neural Net for segmentation


class SegmentationNeuralNet(AbstractNeuralNet):
    """
    Segmentation Neural Net 
    """

    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0,
        view_freq=1
        ):
        """
        Initialization
            -patchproject (str): path project
            -nameproject (str):  name project
            -no_cuda (bool): system cuda (default is True)
            -parallel (bool)
            -seed (int)
            -print_freq (int)
            -gpu (int)
            -view_freq (in epochs)
        """

        super(SegmentationNeuralNet, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu  )
        self.view_freq = view_freq

 
    def create(self, 
        arch, 
        num_output_channels, 
        num_input_channels,  
        loss, 
        lr, 
        momentum, 
        optimizer, 
        lrsch,          
        pretrained=False,
        size_input=388,

        ):
        """
        Create            
            -arch (string): architecture
            -loss (string):
            -lr (float): learning rate
            -optimizer (string) : 
            -lrsch (string): scheduler learning rate
            -pretrained (bool)
        """
        super(SegmentationNeuralNet, self).create( arch, num_output_channels, num_input_channels, loss, lr, momentum, optimizer, lrsch, pretrained)
        self.size_input = size_input
        
        self.accuracy = nloss.Accuracy()
        self.dice = nloss.Dice()
       
        # Set the graphic visualization
        self.logger_train = Logger( 'Train', ['loss'], ['accs', 'dices'], self.plotter  )
        self.logger_val   = Logger( 'Val  ', ['loss'], ['accs', 'dices'], self.plotter )

        self.visheatmap = gph.HeatMapVisdom(env_name=self.nameproject, heatsize=(100,100) )
        self.visimshow = gph.ImageVisdom(env_name=self.nameproject, imsize=(100,100) )

      
    def training(self, data_loader, epoch=0):
        

        #reset logger
        self.logger_train.reset()
        data_time = AverageMeter()
        batch_time = AverageMeter()

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
                targets = targets.cuda(non_blocking=True)
                inputs_var  = Variable(inputs.cuda(),  requires_grad=False)
                targets_var = Variable(targets.cuda(), requires_grad=False)
                weights_var = Variable(weights.cuda(), requires_grad=False)
            else:
                inputs_var  = Variable(inputs,  requires_grad=False)
                targets_var = Variable(targets, requires_grad=False)
                weights_var = Variable(weights, requires_grad=False)

            # fit (forward)
            outputs = self.net(inputs_var)

            # measure accuracy and record loss
            loss = self.criterion(outputs, targets_var, weights_var)            
            accs = self.accuracy(outputs, targets_var )
            dices = self.dice( outputs, targets_var )
              
            # optimizer
            self.optimizer.zero_grad()
            (loss*batch_size).backward()
            self.optimizer.step()
            
            # update
            self.logger_train.update(
                {'loss': loss.data[0] },
                {'accs': accs, 'dices': dices },      
                batch_size,
                )
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:  
                self.logger_train.logger( epoch, epoch + float(i+1)/len(data_loader), i, len(data_loader), batch_time,   )




    def evaluate(self, data_loader, epoch=0):
        
        # reset loader
        self.logger_val.reset()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, sample in enumerate(data_loader):
                
                # get data (image, label)
                inputs, targets, weights = sample['image'], sample['label'], sample['weight']
                batch_size = inputs.size(0)

                if self.cuda:
                    targets = targets.cuda( non_blocking=True )
                    inputs_var  = Variable(inputs.cuda(),  requires_grad=False, volatile=True)
                    targets_var = Variable(targets.cuda(), requires_grad=False, volatile=True)
                    weights_var = Variable(weights.cuda(), requires_grad=False, volatile=True)
                else:
                    inputs_var  = Variable(inputs,  requires_grad=False, volatile=True)
                    targets_var = Variable(targets, requires_grad=False, volatile=True)
                    weights_var = Variable(weights, requires_grad=False, volatile=True)
                
                # fit (forward)
                outputs = self.net(inputs_var)

                # measure accuracy and record loss
                loss = self.criterion(outputs, targets_var, weights_var)   
                accs = self.accuracy(outputs, targets_var )
                dices = self.dice( outputs, targets_var )  
               

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # update
                self.logger_val.update( 
                    {'loss': loss.data[0] },
                    {'accs': accs, 'dices': dices },      
                    batch_size,          
                    )

                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch, epoch, i,len(data_loader), 
                        batch_time, 
                        bplotter=False,
                        bavg=True, 
                        bsummary=False,
                        )

        #save validation loss
        self.vallosses = self.logger_val.info['loss']['loss'].avg
        acc = self.logger_val.info['metrics']['accs'].avg

        self.logger_val.logger(
            epoch, epoch, i, len(data_loader), 
            batch_time,
            bplotter=True,
            bavg=True, 
            bsummary=True,
            )

        #vizual_freq
        if epoch % self.view_freq == 0:
            
            prob = F.softmax(outputs,dim=1)
            prob = prob.data[0]
            _,maxprob = torch.max(prob,0)
            
            self.visheatmap.show('Label', targets_var.data.cpu()[0].numpy()[1,:,:] )
            self.visheatmap.show('Weight map', weights_var.data.cpu()[0].numpy()[0,:,:])
            self.visheatmap.show('Image', inputs_var.data.cpu()[0].numpy()[0,:,:])
            self.visheatmap.show('Max prob',maxprob.cpu().numpy().astype(np.float32) )
            for k in range(prob.shape[0]):                
                self.visheatmap.show('Heat map {}'.format(k), prob.cpu()[k].numpy() )
                
        
        return acc

    def test(self, data_loader, bgt=False):
         
        n = len(data_loader)*data_loader.batch_size
        Yhat = np.zeros((n, self.num_output_channels ))
        Y = np.zeros((n,) )
        Ids = np.zeros((n,) )
        k=0

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, sample in enumerate( tqdm(data_loader) ):
                
                # get data (image, label)
                inputs  = sample['image']      
                targets = sample['label'] if bgt else np.zeros( (inputs.shape) )                     
                Id = sample['id'] if not bgt else np.zeros( (inputs.shape[0]) ) 
                
                x = inputs.cuda() if self.cuda else inputs    
                x  = Variable(x, requires_grad=False, volatile=True )
                
                # fit (forward)
                yhat = self.net(x)
                yhat = F.softmax(yhat, dim=1)    
                yhat = nutl.to_np(yhat)
    
                for j in range(yhat.shape[0]):
                        Y[k] = targets[j]
                        Yhat[k,:] = yhat[j]
                        Ids[k] = Id[j]  
                        k+=1 

                #print( 'Test:', i , flush=True )

        Yhat = Yhat[:k,:]
        Y = Y[:k]
        Ids = Ids[:k]
                
        return Ids, Yhat, Y

    
    
    def inference(self, image):        
        
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image    
            x  = Variable(x, requires_grad=False, volatile=True )
            msoft = nn.Softmax()
            yhat = msoft( self.net(x) )
            yhat = nutl.to_np(yhat).transpose(2,3,1,0)[...,0]

        return yhat


    def representation(self, data_loader):
        """"
        Representation
            -data_loader: simple data loader for image
        """
                
        # switch to evaluate mode
        self.net.eval()

        n = len(data_loader)*data_loader.batch_size
        k=0

        # embebed features 
        embX = np.zeros([n,self.net.dim])
        embY = np.zeros([n,1])

        batch_time = nutl.AverageMeter()
        end = time.time()
        for i, sample in enumerate(data_loader):
                        
            # get data (image, label)
            inputs, targets = sample['image'], nutl.argmax(sample['labels'])
            inputs_var = nutl.to_var(inputs, self.cuda, False, True )

            # representation
            emb = self.net.representation(inputs_var)
            emb = nutl.to_np(emb)
            for j in range(emb.shape[0]):
                embX[k,:] = emb[j,:]
                embY[k] = targets[j]
                k+=1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Representation: |{:06d}/{:06d}||{batch_time.val:.3f} ({batch_time.avg:.3f})|'.format(i,len(data_loader), batch_time=batch_time) )


        embX = embX[:k,:]
        embY = embY[:k]


        return embX, embY
    
    def _create_model(self, arch, num_output_channels, num_input_channels, pretrained ):
        """
        Create model
            -arch (string): select architecture
            -num_classes (int)
            -num_channels (int)
            -pretrained (bool)
        """    

        self.net = None    

        #-------------------------------------------------------------------------------------------- 
        # select architecture
        #--------------------------------------------------------------------------------------------
        #kw = {'num_classes': num_output_channels, 'num_channels': num_input_channels, 'pretrained': pretrained}

        if arch == 'unet':
            self.net = nnmodels.unet( num_classes = num_output_channels )  
        elif arch == 'simpletsegnet':
            self.net = nnmodels.simpletsegnet( num_classes = num_output_channels, num_channels=num_input_channels )  
        elif arch == 'unet11':
            self.net = nnmodels.unet11( num_classes = num_output_channels ) 
        elif arch == 'dunet':
            self.net = nnmodels.dunet( n_classes = num_output_channels )                   
        else:
            assert(False)
        
        self.s_arch = arch
        self.num_output_channels = num_output_channels
        self.num_input_channels = num_input_channels

        if self.cuda == True:
            self.net.cuda()
        if self.parallel == True and self.cuda == True:
            self.net = nn.DataParallel(self.net, device_ids= range( torch.cuda.device_count() ))

    def _create_loss(self, loss):

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

        self.s_loss = loss





