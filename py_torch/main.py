from contextlib import ExitStack
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

# assign objects to groups (classes)
GROUPS = [
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
    transformation = Transformation(opt)
    item_transform = transformation.item_transform

    with ExitStack() as es:
        if opt.stream:
            # Launch Blender instance. Upon exit of this script all Blender instances will be closed.
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
            shuffle = False  # does not support shuffle
            ds.stream_length(4)
            if opt.record:
                ds.enable_recording(opt.record_path)
        elif opt.replay:
            ds = btt.FileDataset(opt.train_path, item_transform=item_filter(item_transform, 
                opt.bbox_visibility_threshold))
            shuffle = True
        else:
            ds = TLessTrainDataset(opt.train_path, item_transform)
            shuffle = True

        num_samples = len(ds)
        # num_samples = 1000  # for debugging
        logging.info(f"Dataset: {ds.__class__.__name__}")
        logging.info(f"Num. of samples: {num_samples}")
        logging.info(f"Batch size: {opt.batch_size}")

        split = int(0.9 * num_samples)
        logging.info(f"Split data set at sample nr. {split}")
        train_ds = data.Subset(ds, indices=list(range(split)))
        val_ds = data.Subset(ds, indices=list(range(split, num_samples)))

        logging.info(f"Training data size: {len(train_ds)}")
        logging.info(f"Validation data size: {len(val_ds)}")

        train_dl = data.DataLoader(train_ds, batch_size=opt.batch_size, 
            num_workers=opt.worker_instances, shuffle=shuffle)
        val_dl = data.DataLoader(val_ds, batch_size=opt.batch_size, 
            num_workers=opt.worker_instances, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"On device: {device}")

        # head_name: num. of channels
        heads = {"cpt_hm": opt.num_classes, "cpt_off": 2, "wh": 2}
        model = get_model(heads)
        model.to(device)

        loss_fn = CenterLoss()

        optimizer = optim.Adam(model.parameters(), 
            lr=opt.lr, weight_decay=opt.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
            step_size=opt.lr_step_size, gamma=0.2)

        num_params = sum(p.numel() for p in model.parameters())
        print(f'Model with: {num_params/10**6:.2f}M number of parameters')

        if opt.resume:
            print(f'load checkpoint from {opt.model_path_to_load}...')
            checkpoint = torch.load(opt.model_path_to_load)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] +1 
            best_metric = checkpoint['best_metric']
            best_loss = checkpoint['best_loss']
        else:
            print('train from scratch...')
            start_epoch = 0
            best_metric = 0  # higher = better
            best_loss = 1000  # lower = better

        if torch.cuda.device_count() > 1:  # default use all GPUs
            print("Use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)  # device_ids

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

if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)
    opt = Config("./configs/config.txt")
    print(opt)

    main(opt)
