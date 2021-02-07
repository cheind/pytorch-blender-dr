from contextlib import ExitStack
import albumentations as A
from blendtorch.btt import constants
import numpy as np
import logging
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import argparse

# git clone https://github.com/cheind/pytorch-blender.git <DST>
# pip install -e <DST>/pkg_pytorch
from blendtorch import btt

# relative imports, call with -m option
# >> python -m py_torch.stream
from .utils import Config
from .train import (stream_train, stream_eval, add_cdistr,
    use_multiple_devices, save_checkpoint, save_best_performing_checkpoint)
from .loss import CenterLoss
from .model import get_model
from .visu import iterate  # iterate over dataloader (debugging)
from .transform import Transform
from .constants import GROUPS

def main(opt):

    if opt.debug:
        logging.basicConfig(level=logging.DEBUG)
        opt.num_batches = 128
        opt.loss_threshold = 7.0
        opt.max_level_len = 32
        opt.num_objects = 7 
        opt.objects_step = 1
        opt.val_interval = 64
        opt.val_len = 16 
        opt.save_interval = 128 
        opt.train_vis_interval = 32
        opt.val_vis_interval = 8 

    launch_info = btt.LaunchInfo.load_json(opt.launch_info)
    data_addr = launch_info.addresses['DATA']
    # CTRL is a list of addresses for establishing connection
    # to multiple Blender instances
    ctrl_addr = launch_info.addresses['CTRL']  

    # Bidirectional communication with this pytorch training script
    # from Blender instance(s) 
    remotes = [btt.DuplexChannel(addr) for addr in ctrl_addr]

    num_objects = opt.num_objects
    # start with uniform class distribution 1-30 classes
    # and will later update the distribution to 
    # populate the scene with focus of objects we classified weakly
    cdistr = {k:1 for k in range(30)}  # 1-30 -> 0-29

    for remote in remotes:
        remote.send(num_objects=opt.num_objects, cdistr=cdistr)

    # Choose augmentations for training
    augmentations = [
        A.HueSaturationValue(hue_shift_limit=20, 
            sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.ChannelShuffle(p=0.5),
        A.HorizontalFlip(p=0.2),
    ]
    # Resize images to opt.h, opt.w in training and
    # filter bboxes with opt.bbox_visibility_threshold
    transformation = Transform(opt, augmentations=augmentations, 
        filter=True, resize_crop=True)

    # number of samples to produce 
    max_items = opt.batch_size * (opt.num_batches + 
        opt.val_len * int(opt.num_batches / opt.val_interval + 1)) 
    max_batches = max_items / opt.batch_size
    logging.info(f'Total samples: {max_items} (or {max_batches} batches)')
    logging.info(f"Number of total training batches: {opt.num_batches}")
    logging.info(f"Batch size: {opt.batch_size}")
    
    # btt.RemoteIterableDataset does not support shuffle
    ds = btt.RemoteIterableDataset(
        data_addr,
        max_items=max_items, 
        item_transform=transformation,
    )

    # Setup DataLoader
    dl = data.DataLoader(ds, batch_size=opt.batch_size, 
        num_workers=opt.worker_instances, shuffle=False)

    loss_fn = CenterLoss()

    if opt.pretrained:
        logging.info('Train with pretrained backbone...')
    else:
        logging.info('Train with randomly initialized backbone...')

    # head_name: num. of channels
    heads = {"cpt_hm": opt.num_classes, "cpt_off": 2, "wh": 2}
    model = get_model(heads, pretrained=opt.pretrained)

    # export CUDA_VISIBLE_DEVICES=1,2,3 or 1 or 0,1
    # Run cuda.py to see available GPUs or nvidia-smi
    # Note: if CUDA_VISIBLE_DEVICES=1,2,3 then cuda device 0
    # in pytorch refers to physical cuda device 1
    device = torch.device('cuda' if torch.cuda.is_available() and opt.use_cuda else 'cpu')
    logging.info(f'Using device: {device}')
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model with: {num_params/10**6:.2f}M number of parameters')

    optimizer = optim.Adam(model.parameters(), 
        lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
        step_size=opt.lr_step_size, gamma=0.2)

    best_metric = 0  # higher = better
    best_loss = 1000  # lower = better

    logging.info('Open tensorboard to track trainings progress')
    # tensorboard --logdir=runs --bind_all
    writer = SummaryWriter()

    train_meter = MetricMeter()
    eval_meter = MetricMeter()               

    steps_since_adjustment = 0
    
    logging.info(f'Start training with data stream(s) from Blender')
    while batch_count < max_batches:
        logging.info(f'Progress: {batch_count}/{max_batches}')

        # train till opt.val_interval batches are processed
        batch_count = stream_train(train_meter, batch_count, model, optimizer, 
            dl, loss_fn, writer, opt)
        train_meter.reset()
        scheduler.step()

        if not isinstance(model, nn.DataParallel):
            state_dict = model.state_dict() 
        else:
            state_dict = model.module.state_dict()

        save_dict = {
            'batch': batch_count,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_metric': best_metric,
            'best_loss': best_loss,
        }

        # save model, optimizer, scheduler... (regular checkpoints)
        save_checkpoint(save_dict, batch_count, opt)
        
        # perform a evaluation step for opt.val_len batches
        batch_count, prec = stream_eval(eval_meter, batch_count, model,  
            dl, loss_fn, writer, opt)
        
        # save model, optimizer, scheduler... (best performing checkpoint)
        best_metric, best_loss = save_best_performing_checkpoint(save_dict, best_metric, 
            best_loss, eval_meter, opt, use_loss=False, use_metric=True)
        eval_meter.reset()

        steps_since_adjustment += 1
        # increase or decrease difficulty by stream adjustment
        steps_since_adjustment = adjust_stream(loss, prec, cdistr, 
            steps_since_adjustment, batch_count, remotes, opt)

if __name__ == '__main__':
    config_path = './configs'
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='stream_config.txt')
    args = parser.parse_args()

    opt = Config(f'{config_path}/{args.config}')
    main(opt)
