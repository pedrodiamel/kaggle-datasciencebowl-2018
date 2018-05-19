

import os
import math
import shutil
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
from tqdm import tqdm

from . import netmodels as nnmodels
from . import netlosses as nloss

from pytvision.neuralnet import AbstractNeuralNet
from pytvision.logger import Logger, AverageFilterMeter, AverageMeter
from pytvision import graphic as gph
from pytvision import netlearningrate
from pytvision import utils as pytutils

#----------------------------------------------------------------------------------------------
# Neural Net for Segmentation


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
                yhat = pytutils.to_np(yhat)
    
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
            yhat = pytutils.to_np(yhat).transpose(2,3,1,0)[...,0]

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

        batch_time = AverageMeter()
        end = time.time()
        for i, sample in enumerate(data_loader):
                        
            # get data (image, label)
            inputs, targets = sample['image'], pytutils.argmax(sample['labels'])
            inputs_var = pytutils.to_var(inputs, self.cuda, False, True )

            # representation
            emb = self.net.representation(inputs_var)
            emb = pytutils.to_np(emb)
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





