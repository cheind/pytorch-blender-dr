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
# >> python -m py_torch.main
from .utils import Config
from .train import (train, eval, use_multiple_devices, 
    save_checkpoint)
from .loss import CenterLoss
from .model import get_model
from .visu import iterate  # iterate over dataloader (debugging)
from .transform import Transform
from .evaluation import evaluate_model
from .utils import MetricMeter
from .coco_report import ap_values
from .dataset import TLessDataset

def main(opt):
            
    def load_from_checkpoint(opt):
        if opt.resume or not opt.train:
            nonlocal model
            logging.info(f'Load checkpoint from {opt.model_path_to_load}...')
            checkpoint = torch.load(opt.model_path_to_load, 
                map_location='cuda' if opt.use_cuda else 'cpu')
            model.load_state_dict(checkpoint['model'])
            if opt.train:  # skipt this for inference
                nonlocal optimizer, scheduler
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])

            start_epoch = checkpoint.get('epoch', -1) + 1         
            best_metric = checkpoint.get('best_metric', 0)
            best_loss = checkpoint.get('best_loss', 1000)
        
            logging.info(f'Start epoch: {start_epoch}')
            logging.info(f'Best loss: {best_loss}')
            logging.info(f'Best metric: {best_metric}')
        else: 
            logging.info('Train with pretrained backbone...')
            start_epoch = 0
            best_metric = 0  # higher = better
            best_loss = 1000  # lower = better
        
        return start_epoch, best_metric, best_loss

    # Choose augmentations for training
    augmentations = [
        A.HueSaturationValue(hue_shift_limit=20, 
            sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.ChannelShuffle(p=0.5),
        A.HorizontalFlip(p=0.2),
    ] if opt.train else []
    # Resize images to opt.h, opt.w in training
    transformation = Transform(opt, augmentations=augmentations, 
        filter=opt.replay, resize_crop=opt.train)

    with ExitStack() as es:
        if opt.replay:
            logging.info('Use data from Blender replay...')
            ds = btt.FileDataset(
                opt.train_path, 
                item_transform=transformation,
            )
        else:
            logging.info('Use BOP data...')
            ds = TLessDataset(opt, transformation)

        # head_name: num. of channels
        heads = {"cpt_hm": opt.num_classes, "cpt_off": 2, "wh": 2}
        model = get_model(heads)

        # export CUDA_VISIBLE_DEVICES=1,2,3 or 1 or 0,1
        # Run cuda.py to see available GPUs or nvidia-smi
        # Note: if CUDA_VISIBLE_DEVICES=1,2,3 then cuda device 0
        # in pytorch refers to physical cuda device 1
        device = torch.device('cuda' if torch.cuda.is_available() and opt.use_cuda else 'cpu')
        logging.info(f'Using device: {device}')
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f'Model with: {num_params/10**6:.2f}M number of parameters')
            
        # either we load model (optimizer, scheduler) 
        # or train from pretrained backbone
        if opt.train:
            optimizer = optim.Adam(model.parameters(), 
                    lr=opt.lr, weight_decay=opt.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                step_size=opt.lr_step_size, gamma=0.2)
        
        start_epoch, best_metric, best_loss = load_from_checkpoint(opt)

        if not opt.train:  
            logging.info("Runnig script for inference...")
            if opt.debug:  # sanity check on subset
                ds = data.Subset(ds, indices=list(range(min(100, len(ds)))))
            test_dl = data.DataLoader(ds, batch_size=1, 
                num_workers=opt.worker_instances, shuffle=False)
            evaluate_model(model, test_dl, opt)
        else:
            logging.info("Runnig script for training...")       
            
            use_multiple_devices(model, opt)
            loss_fn = CenterLoss()
            
            num_samples = len(ds) if not opt.debug else min(100, len(ds))
            logging.info(f"Number of samples: {num_samples}")
            logging.info(f"Batch size: {opt.batch_size}")

            split = int(0.9 * num_samples)
            logging.info(f"Split data set at sample nr. {split}")
            train_ds = data.Subset(ds, indices=list(range(split)))
            val_ds = data.Subset(ds, indices=list(range(split, num_samples)))

            logging.info(f"Training data size: {len(train_ds)}")
            logging.info(f"Validation data size: {len(val_ds)}")

            train_dl = data.DataLoader(train_ds, batch_size=opt.batch_size, 
                num_workers=opt.worker_instances, shuffle=True, drop_last=True)
            val_dl = data.DataLoader(val_ds, batch_size=opt.batch_size, 
                num_workers=opt.worker_instances, shuffle=False)

            logging.info('Open tensorboard to track trainings progress')
            # tensorboard --logdir=runs --bind_all
            writer = SummaryWriter()

            train_meter = MetricMeter()
            eval_meter = MetricMeter()     

            for epoch in range(start_epoch, start_epoch + opt.num_epochs):
                logging.info(f"Epoch: {epoch} / {start_epoch + opt.num_epochs - 1}")
                train(train_meter, epoch, model, optimizer, 
                    train_dl, loss_fn, writer, opt)
                scheduler.step()

                prec = eval(eval_meter, epoch, model, val_dl, loss_fn, writer, opt)
                
                if not isinstance(model, nn.DataParallel):
                    state_dict = model.state_dict() 
                else:
                    state_dict = model.module.state_dict()
                save_dict = {
                    'epoch': epoch,
                    'model': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }

                loss = eval_meter.get_avg('total_loss')
                metric = ap_values(prec).mean()  # mAP
                # set x-axis of metric tracking to same as loss for better graph comparison
                writer.add_scalar('Val/metric', metric, epoch * len(val_dl))

                # save model, optimizer, scheduler... 
                best_metric, best_loss = save_checkpoint(save_dict, 
                    epoch, metric, loss, best_metric, best_loss, opt)

if __name__ == '__main__':
    config_path = './configs'
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.txt')
    args = parser.parse_args()

    opt = Config(f'{config_path}/{args.config}')

    if opt.debug:
        opt.model_score_threshold = 0.1
        opt.num_epochs = 1
        opt.train_vis_interval = 1
        opt.val_vis_interval = 1

    main(opt)
