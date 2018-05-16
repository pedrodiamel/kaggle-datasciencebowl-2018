
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn

from torchlib.datasets import dsxbdata 
from torchlib.datasets import dsxbtransform as dsxbtrans
from torchlib import neuralnet as nnet

from argparse import ArgumentParser
import datetime

def arg_parser():
    """Arg parser"""    
    parser = ArgumentParser()
    parser.add_argument('data', metavar='DIR', 
                        help='path to dataset')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('-g', '--gpu', default=0, type=int, metavar='N',
                        help='divice number (default: 0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', 
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--snapshot', '-sh', default=10, type=int, metavar='N',
                        help='snapshot (default: 10)')
    parser.add_argument('--project', default='./runs', type=str, metavar='PATH',
                        help='path to project (default: ./runs)')
    parser.add_argument('--name', default='exp', type=str,
                        help='name of experiment')
    parser.add_argument('--resume', default='model_best.pth.tar', type=str, metavar='NAME',
                    help='name to latest checkpoint (default: none)')
    parser.add_argument('--arch', default='simplenet', type=str,
                        help='architecture')
    parser.add_argument('--finetuning', action='store_true', default=False,
                    help='Finetuning')
    parser.add_argument('--loss', default='cross', type=str,
                        help='loss function')
    parser.add_argument('--opt', default='adam', type=str,
                        help='optimize function')
    parser.add_argument('--scheduler', default='fixed', type=str,
                        help='scheduler function for learning rate')
    parser.add_argument('--image-size', default=388, type=int, metavar='N',
                        help='image size')
    parser.add_argument('--parallel', action='store_true', default=False,
                    help='Parallel')
    return parser



def main():
    
    # parameters
    parser = arg_parser();
    args = parser.parse_args();
    imsize = args.image_size;
    parallel=args.parallel;

    network = nnet.Network(
        patchproject=args.project,
        nameproject=args.name,
        no_cuda=args.no_cuda,
        parallel=parallel,
        seed=args.seed,
        print_freq=args.print_freq,
        gpu=args.gpu
        )
        
    network.create( 
        arch=args.arch, 
        num_classes=3, 
        loss=args.loss, 
        lr=args.lr, 
        momentum=args.momentum,
        opt=args.opt,
        lrsch=args.scheduler,
        pretrained=args.finetuning
        )
    
    # resume
    network.resume( os.path.join(network.pathmodels, args.resume ) )
    cudnn.benchmark = True

    # datasets
    # training dataset
    train_data = dsxbdata.DSXBDataset(
        args.data, 
        dsxbdata.train, 
        transform=transforms.Compose([
            #dsxbtrans.ElasticDistort(size_grid=50, deform=15),
            dsxbtrans.GeometricDistort(angle=360, translation=0.2, warp=0.0),
            dsxbtrans.RandomCrop( cropsize=(250,250) ),
            dsxbtrans.ColorDistort(),
            dsxbtrans.UnetResize(imsize=imsize),            
            dsxbtrans.ToTensor(),
            dsxbtrans.ElasticTorchDistort(size_grid=10, deform=0.05),
            dsxbtrans.Normalize(), 
            ])
        )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, pin_memory=network.cuda, drop_last=True )
    
    # validate dataset
    val_data = dsxbdata.DSXBDataset(
        args.data, 
        dsxbdata.test, 
        transform=transforms.Compose([
            #dsxbtrans.ElasticDistort(size_grid=50, deform=15),
            dsxbtrans.GeometricDistort(angle=360, translation=0.2, warp=0.0),
            dsxbtrans.RandomCrop( cropsize=(250,250) ),
            dsxbtrans.ColorDistort(),
            dsxbtrans.UnetResize(imsize=imsize),            
            dsxbtrans.ToTensor(),
            dsxbtrans.ElasticTorchDistort(size_grid=10, deform=0.05),
            dsxbtrans.Normalize(), 
            ])
        )

    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, pin_memory=network.cuda, drop_last=False)
       
    # print neural net class
    print('SEGTorch: {}'.format(datetime.datetime.now()) )
    print(network)

    # initialization evaluate
    network.evaluate(val_loader, epoch=network.start_epoch)


    best_prec = 0
    for epoch in range(network.start_epoch, args.epochs):       

        network.adjust_learning_rate(epoch)     
        network.training(train_loader, epoch)

        print('Epoch: {}/{}'.format(epoch,args.epochs))

        prec = network.evaluate(val_loader, epoch+1)            

        # remember best prec@1 and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)

        if epoch % args.snapshot == 0 or is_best:
            network.save(epoch, best_prec, is_best, 'chk{:06d}.pth.tar'.format(epoch))
        
        
               
    print("Optimization Finished!")
    print("DONE!!!")



if __name__ == '__main__':
    main()