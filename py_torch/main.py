from contextlib import ExitStack
from albumentations import augmentations
import numpy as np
import logging
import os
import sys
import cv2
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

# git clone https://github.com/cheind/pytorch-blender.git <DST>
# pip install -e <DST>/pkg_pytorch
from blendtorch import btt

from .utils import Config, generate_heatmap
from .train import train, eval
from .loss import CenterLoss
from .model import get_model
from .decode import decode, filter_dets
from .visu import render, iterate
from .transform import Transformation, item_filter
from .evaluation import create_gt_anns, evaluate

GROUPS = [  # assign objects to groups (classes)
    [1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12,],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30],
]
NAMES = [f"{i}" for i in range(len(GROUPS))]
CATEGORIES = [{"id": id, "name": name} for id, name in enumerate(NAMES)]
# to map the 1 to 30 original classes to new classes of 0 to 5
MAPPING = {old_cls_id: new_cls_id for new_cls_id, group in enumerate(GROUPS) 
                                        for old_cls_id in group}
            
def _to_float(x):
    return float(f"{x:.2f}")

def main(opt):
    if opt.train:
        logging.info("Runnig script for training...")
        augmentation = [
            A.HueSaturationValue(hue_shift_limit=20, 
                sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.ChannelShuffle(p=0.5),
            A.HorizontalFlip(p=0.2),
        ])
    else: 
        logging.info("Runnig script for inference...")
        augmentation = []
        
    transformation = Transformation(opt, augmentations=augmentation)

    # TODO: stream, replay

    with ExitStack() as es:
        if opt.stream:
            logging.info('Stream data from Blender instance(s)...')
            # Launch Blender instance. 
            # Upon exit of this script all Blender instances will 
            # be closed.
            bl = es.enter_context(
                btt.BlenderLauncher(
                    scene=f"{opt.scene}.blend",
                    script=f"{opt.scene}.blend.py",
                    num_instances=opt.blender_instances,
                    named_sockets=['DATA'],
                    blend_path=opt.blend_path,
                )
            )
            ds = btt.RemoteIterableDataset(
                bl.launch_info.addresses['DATA'],
                item_transform=item_filter(item_transform, 
                    opt.bbox_visibility_threshold)
            )
            # shuffle = False  # does not support shuffle
            ds.stream_length(4)
            if opt.record:
                ds.enable_recording(opt.record_path)
        elif opt.replay:
            logging.info('Use data from Blender replay...')
            ds = btt.FileDataset(opt.train_path, item_transform=item_filter(item_transform, 
                opt.bbox_visibility_threshold))
            # shuffle = True
        else:
            logging.info('Use saved data from data folders...')
            ds = TLessDataset(opt, transformation)

        # head_name: num. of channels
        heads = {"cpt_hm": opt.num_classes, "cpt_off": 2, "wh": 2}
        model = get_model(heads)
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f'Model with: {num_params/10**6:.2f}M number of parameters')

        if opt.resume or not opt.train:
            logging.info(f'Load checkpoint from {opt.model_path_to_load}...')
            checkpoint = torch.load(opt.model_path_to_load)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] +1 
            best_metric = checkpoint['best_metric']
            best_loss = checkpoint['best_loss']

            logging.info(f'Start epoch: {start_epoch}')
            logging.info(f'Best loss: {best_loss}')
            logging.info(f'Best metric: {best_metric}')
        else:
            logging.info('Train from scratch...')
            start_epoch = 0
            best_metric = 0  # higher = better
            best_loss = 1000  # lower = better
   
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if opt.train:
            loss_fn = CenterLoss()
            optimizer = optim.Adam(model.parameters(), 
                lr=opt.lr, weight_decay=opt.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                step_size=opt.lr_step_size, gamma=0.2)

            if torch.cuda.device_count() > 1:  # default use all GPUs
                print("Use", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)  # device_ids to select GPUs

            num_samples = 1000  # len(ds) TODO
            logging.info(f"Number of samples: {num_samples}")
            logging.info(f"Batch size: {batch_size}")

            split = int(0.9 * num_samples)
            logging.info(f"Split data set at sample nr. {split}")
            train_ds = data.Subset(ds, indices=list(range(split)))
            val_ds = data.Subset(ds, indices=list(range(split, num_samples)))

            logging.info(f"Training data size: {len(train_ds)}")
            logging.info(f"Validation data size: {len(val_ds)}")

            train_dl = data.DataLoader(train_ds, batch_size=opt.batch_size, 
                num_workers=opt.worker_instances, shuffle=True)
            val_dl = data.DataLoader(val_ds, batch_size=opt.batch_size, 
                num_workers=opt.worker_instances, shuffle=False)

            logging.info('Open tensorboard to track trainings progress')
            # tensorboard --logdir=runs --bind_all
            writer = SummaryWriter()

            for epoch in range(start_epoch, opt.num_epochs + start_epoch):
                logging.info(f"Epoch: {epoch} / {start_epoch + opt.num_epochs}")
                _ = train(epoch, model, optimizer, train_dl, loss_fn, writer, opt)
                
                if not isinstance(model, nn.DataParallel):
                    state_dict = model.state_dict() 
                else:
                    state_dict = model.module.state_dict()

                torch.save({
                    'epoch': epoch,
                    'model': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_metric': best_metric,
                    'best_loss': best_loss,
                }, f"{opt.model_folder}/{opt.model_last_tag}.pth")

                if epoch % opt.save_interval == 0 and epoch != 0:
                    torch.save({
                        'epoch': epoch,
                        'model': state_dict,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_metric': best_metric,
                        'best_loss': best_loss,
                    }, f"{opt.model_folder}/{opt.epoch_model_tag}_{epoch}.pth")

                if epoch % opt.val_interval == 0:
                    # TODO: choose best model on mAP metric instead of total loss 
                    meter = eval(epoch, model, val_dl, loss_fn, writer, opt)
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

        else:
            model.eval()
            test_dl = data.DataLoader(ds, batch_size=1, 
                num_workers=opt.worker_instances, shuffle=False)

        
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    opt = Config("./configs/config.txt")

    main(opt)
