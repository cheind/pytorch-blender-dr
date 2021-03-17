import argparse
import math
import time
import os
import sys
import copy
from contextlib import ExitStack
import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

# git clone https://github.com/cheind/pytorch-blender.git <DST>
# pip install -e <DST>/pkg_pytorch
from blendtorch import btt

# relative imports, call with -m option
# >> python -m py_torch.main
from .train import train, val, save_checkpoint
from .loss import CenterLoss
from .visu import iterate  # iterate over dataloader (debugging)
from .transform import Transform
from .evaluation import evaluate_model
from .coco_report import ap_values
from .dataset import TLessDataset
from .utils import (profile, profile_training, init_torch_seeds,
    model_info, time_synchronized, initialize_weights,
    is_parallel, setup, cleanup, MetricMeter, data_mean_and_std,
    Config, prune_weights, item_transform_image_only)

from .model import get_model
#from model_v2 import get_model
#from model_v3 import get_model

def main(rank, opt):
    opt = copy.deepcopy(opt)
    opt.rank = rank
    
    if opt.world_size > 1 and opt.cuda:  
        # must be called before any other cuda related function
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in opt.gpus])
    
    np.random.seed(opt.seed)
    init_torch_seeds(opt.seed, verbose=(rank == 0))
    
    augmentations = [
        A.HueSaturationValue(hue_shift_limit=20, 
            sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.ChannelShuffle(p=0.5),
        A.HorizontalFlip(p=0.2),
    ] if opt.train else []
    
    tf = Transform(opt, augmentations=augmentations, 
        vis_filter=opt.replay, resize_crop=opt.train)

    with ExitStack() as es:
        if opt.replay:
            if opt.rank == 0:
                print('Use data from Blender replay.')
            ds = btt.FileDataset(
                opt.train_path, 
                item_transform=tf if opt.world_size == 1 else tf.item_transform_mp,
            )
        else:
            if opt.rank == 0:
                print('Use BOP data.')
            ds = TLessDataset(opt, tf if opt.world_size == 1 else tf.item_transform_mp)

        heads_dict = {'cpt_hm': opt.classes, 'cpt_off': 2, 'wh': 2}
        model = get_model(heads_dict, pretrained=opt.pretrained)
        initialize_weights(model)
            
        if torch.cuda.is_available() and opt.cuda:
            torch.cuda.manual_seed_all(opt.seed)
            if opt.world_size > 1:
                setup(opt.rank, opt.world_size)
            device = torch.device(opt.rank)
            torch.cuda.set_device(opt.rank)
            print(f'Running on rank {opt.rank}, device {torch.cuda.current_device()}.')
        else:
            device = torch.device('cpu')
            opt.cuda = False
            opt.amp = False
            opt.world_size = 1
            print('Running on CPU.')
            
        model.to(device)
            
        if opt.rank == 0:
            model_info(model)
        
        if opt.rank == 0 and opt.profile:
            profile(model, amp=opt.amp)
            profile_training(model, amp=opt.amp)
        
        if opt.resume or not opt.train:
            print(f'Load checkpoint from {opt.model_path_to_load}.')
            map_location = {'cuda:0': f'cuda:{opt.rank}'} if opt.cuda else 'cpu'
            checkpoint = torch.load(opt.model_path_to_load, map_location)
            model.load_state_dict(checkpoint['model'])
            
        if opt.train:
            optimizer = optim.Adam(model.parameters(), 
                    lr=opt.lr, weight_decay=opt.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                step_size=opt.lr_step_size, gamma=opt.lr_step_gamma) 
            scaler = GradScaler(enabled=opt.amp)
            
            if opt.resume:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                scaler.load_state_dict(checkpoint['scaler'])
                
                start_epoch = checkpoint.get('epoch', -1) + 1         
                best_metric = checkpoint.get('best_metric', 0)
                best_loss = checkpoint.get('best_loss', 1E8)
        
                if opt.rank == 0:
                    print(f'Start epoch: {start_epoch}')
                    print(f'Best loss: {best_loss}')
                    print(f'Best metric: {best_metric}')          
            else: 
                if opt.rank == 0:
                    print('Use randomly initialized network...')
                start_epoch = 0
                best_metric = 0  # higher = better
                best_loss = 1E8  # lower = better
            
        if opt.world_size > 1:
            if opt.sync_bn:  # batch statistics across multipe devices
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[opt.rank], find_unused_parameters=True)

        if not opt.train:  
            print('Runnig script for testing...')
            if opt.debug:  # sanity check on subset
                indices = list(range(min(opt.debug_size_test, len(ds))))
                ds = data.Subset(ds, indices=indices)
                
            test_dl = data.DataLoader(ds, batch_size=1, 
                num_workers=opt.workers, shuffle=False)

            # prune lowest model weights to requested sparsity
            if opt.test_sparsity > 0:
                prune_weights(model, amount=opt.test_sparsity)
            
            evaluate_model(model, test_dl, opt)
        else:
            # network training 
            if not opt.debug:
                split = int(0.9 * len(ds))
                train_ds = data.Subset(ds, indices=list(range(split)))
                val_ds = data.Subset(ds, indices=list(range(split, len(ds))))
            else:  # sanity check on subset when debugging
                train_ds = data.Subset(ds, indices=list(range(opt.debug_size_train)))
                val_ds = data.Subset(ds, 
                    indices=list(range(len(ds) - opt.debug_size_val, len(ds))))
                
            if opt.world_size > 1:
                # sampler will split up dataset and each process will 
                # get a part of the whole data exclusively 
                sampler = DistributedSampler(train_ds, shuffle=True) 
            else: 
                sampler = None

            train_dl = data.DataLoader(train_ds, batch_size=opt.batch_size, 
                num_workers=opt.workers, shuffle=(sampler is None), 
                sampler=sampler, drop_last=True, 
                pin_memory=True, persistent_workers=False)
            
            val_dl = data.DataLoader(val_ds, batch_size=opt.batch_size, 
                num_workers=opt.workers, shuffle=False, sampler=None,
                pin_memory=True, persistent_workers=False)
            
            if opt.rank == 0:
                print('Runnig script for training...')
                print(f'Batch size: {opt.batch_size}')
                print(f'Training data size: {len(train_ds)}')
                print(f'Validation data size: {len(val_ds)}')
                print('Open tensorboard to track trainings progress.')
                print('>> tensorboard --logdir=runs --bind_all')
                writer = SummaryWriter()
            else: 
                writer = None 

            train_meter = MetricMeter()
            eval_meter = MetricMeter()    
            
            loss_fn = CenterLoss()
            
            if opt.rank == 0:  # measure overall trainings time
                since = time_synchronized()
                
            opt.start_epoch = start_epoch
            for epoch in range(start_epoch, start_epoch + opt.epochs):
                if opt.world_size > 1:  # so shuffle works properly in DDP mode
                    sampler.set_epoch(epoch)
                
                train(train_meter, epoch, model, optimizer, 
                    scaler, train_dl, loss_fn, writer, opt)
                
                scheduler.step()
                
                if opt.rank == 0:
                    prec = val(eval_meter, epoch, model, val_dl, loss_fn, writer, opt)
                    
                    if prec is not None:
                        metric = ap_values(prec).mean()  # mAP
                    else:
                        metric = 0
                    writer.add_scalar('Val/metric', metric, epoch * len(val_dl))

                    loss = eval_meter.get_avg('total_loss')
                    msd = (model.module.state_dict() if is_parallel(model) 
                        else model.state_dict())
                    save_dict = {
                        'epoch': epoch,
                        'model': msd,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                    }
                    best_metric, best_loss = save_checkpoint(save_dict, 
                        epoch, metric, loss, best_metric, best_loss, opt)
            
            if opt.rank == 0:
                elapsed = time_synchronized() - since
                print(f'Training complete: {elapsed // 60:.0f}m {elapsed % 60:.0f}s')
                print(f'Best validation metric/loss: {best_metric:3f}/{best_loss:3f}')
        
            if opt.world_size > 1: 
                cleanup()
                
def run(main_fn, opt):
    mp.spawn(main_fn,
             args=(opt,),
             nprocs=opt.world_size,
             join=True)

if __name__ == '__main__':
    config_path = './configs'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.txt')
    args = parser.parse_args()

    opt = Config(f'{config_path}/{args.config}')
    
    opt.world_size = len(opt.gpus)
        
    if opt.debug:
        # when there is a warning about optimizer step and lr schduler
        # that is totally fine and nothing to worry...
        opt.model_score_threshold_high = 0.1
        opt.epochs = 2
        opt.train_vis_interval = 1
        opt.val_vis_interval = 1
        # how many samples to be processed for debbuging purpose
        opt.debug_size_train = 3 * opt.batch_size * opt.world_size
        opt.debug_size_val = 2 * opt.batch_size
        opt.debug_size_test = 4 * opt.batch_size

    if opt.load_data_stats:
        if opt.replay:
            ds = btt.FileDataset(
                opt.train_path, 
                item_transform=item_transform_image_only,
            )
        else:
            ds = TLessDataset(opt, item_transform_image_only)

        # load custom dataset statistic if available
        mode = 'train' if opt.train else 'test'
        path = f'./etc/dataset_stats_{mode}.npz'

        dl = data.DataLoader(ds, batch_size=1, num_workers=8, shuffle=False)
        if not os.path.exists(path):
            mean, std = data_mean_and_std(dl, channel_first=False)
            np.savez(path, mean=mean.numpy(), std=std.numpy())
            print(f'Saved data statistics to: {path}')
        
        if os.path.exists(path):
            npzfile = np.load(path)
            opt.mean, opt.std = npzfile['mean'], npzfile['std']
            print(f'Loaded data statistics from: {path}')
            print('mean =', opt.mean, 'std =', opt.std)
    
    if opt.world_size > 1 and opt.train:
        run(main, opt)
    else:  # testing without DDP
        opt.world_size = 1
        main(rank=0, opt=opt)
        
    """
    Model Summary: 310 layers, 17.9M parameters, 17.9M gradients, 61.8 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 14.580ms (cuda)
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 46.818ms (cuda)
    Backward time: 313.878ms (cuda)
    Maximum of managed memory: 11.507073024GB
    """
