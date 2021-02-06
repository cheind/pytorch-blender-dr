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
from .train import train, eval, use_multiple_devices
from .loss import CenterLoss
from .model import get_model
from .visu import iterate  # iterate over dataloader (debugging)
from .transform import Transform
from .evaluation import evaluate_model
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
            start_epoch = checkpoint['epoch'] + 1         
            best_metric = checkpoint['best_metric']
            best_loss = checkpoint['best_loss']
        
            logging.info(f'Start epoch: {start_epoch}')
            logging.info(f'Best loss: {best_loss}')
            logging.info(f'Best metric: {best_metric}')
        else: 
            logging.info('Train with pretrained backbone...')
            start_epoch = 0
            best_metric = 0  # higher = better
            best_loss = 1000  # lower = better

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
        
        load_from_checkpoint(opt)

        if not opt.train:  
            logging.info("Runnig script for inference...")
            test_dl = data.DataLoader(ds, batch_size=1, 
                num_workers=opt.worker_instances, shuffle=False)
            evaluate_model(model, test_dl, opt)
        else:
            logging.info("Runnig script for training...")       
            
            use_multiple_devices(model)
            loss_fn = CenterLoss()
            
            num_samples = len(ds) if not opt.debug else 100
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

            stop_epoch = start_epoch + opt.num_epochs
            epochs = range(start_epoch, stop_epoch) if not opt.debug else (0,)
            for epoch in epochs:
                logging.info(f"Epoch: {epoch} / {stop_epoch - 1}")
                _ = train(epoch, model, optimizer, train_dl, loss_fn, writer, opt)
                
                # save model, optimizer, scheduler...
                if not isinstance(model, nn.DataParallel):
                    state_dict = model.state_dict() 
                else:
                    state_dict = model.module.state_dict()

                save_dict = {
                    'model': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_metric': best_metric,
                    'best_loss': best_loss,
                }
                save_checkpoint(save_dict, epoch, opt, stream=False)

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
            
if __name__ == '__main__':
    config_path = './configs'
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.txt')
    args = parser.parse_args()

    opt = Config(f'{config_path}/{args.config}')
    main(opt)
