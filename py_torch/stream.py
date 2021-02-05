from contextlib import ExitStack
import albumentations as A
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
from .train import (stream_train, stream_eval, 
    use_multiple_devices, save_checkpoint)
from .loss import CenterLoss
from .model import get_model
from .visu import iterate  # iterate over dataloader (debugging)
from .transform import Transform

def main(opt):

    if opt.debug:
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
    cls_distr = {k:1 for k in range(30)}  # 1-30 -> 0-29

    for remote in remotes:
        remote.send(num_objects=num_objects, cdistr=cdistr)

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
    logging.info(f'Samples: {max_items} (or {max_batches} batches)')
    
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

    logging.info(f"Number of training batches: {opt.num_batches}")
    logging.info(f"Batch size: {opt.batch_size}")

    logging.info('Open tensorboard to track trainings progress')
    # tensorboard --logdir=runs --bind_all
    writer = SummaryWriter()

    image_id = 0    
    gt_ann_id = 0

    logging.info(f'Start training with data stream(s) from Blender')
    for batch_count in range(opt.num_batches):
        
    # save model, optimizer, scheduler...
    save_checkpoint(model, epoch, opt, stream=True)         

    # TODO: choose best model on mAP metric instead of total loss 
     = eval(meter, batch_count, model, val_dl, loss_fn, writer, opt)
    total_loss = meter.get_avg("total_loss")
    logging.info(f'Evaluation loss: {total_loss}')

    if total_loss <= best_loss:
        best_loss = total_loss
        torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_metric': best_metric,
            'best_loss': best_loss,
        }, f"{opt.model_folder}/{opt.best_model_tag}.pth")

scheduler.step()



if __name__ == '__main__':
    config_path = './configs'
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='stream_config.txt')
    args = parser.parse_args()

    opt = Config(f'{config_path}/{args.config}')
    main(opt)
