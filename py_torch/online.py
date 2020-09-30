from contextlib import ExitStack
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
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

from .utils import Config
from .train import train, eval
from .loss import CenterLoss
from .model import get_model
from .decode import decode, filter_dets
from .visu import render

from tqdm import tqdm
from .utils import MetricMeter
from .train import add_dt, add_gt, add_hms


# group classes
GROUPS = [
    [1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12,],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30],
]
# group names
NAMES = [f"{i}" for i in range(len(GROUPS))]

# each category has a unique id and the category name
CATEGORIES = [{"id": id, "name": name} for id, name in enumerate(NAMES)]

# to map the 1 to 30 class labels to groups from 0 to 5 = new class labels!
CLSES_MAP = {old_cls_id: new_cls_id for new_cls_id, group in enumerate(GROUPS) 
                                        for old_cls_id in group}


class Transformation:

    def __init__(self, opt):
        self.h, self.w = opt.h, opt.w  # e.g. 512 x 512
        self.num_classes = opt.num_classes  # num. of object classes
        self.n_max = opt.n_max  # num. of max objects per image
        self.down_ratio = opt.down_ratio  # => low resolution 128 x 128
        # ImageNet stats
        self.mean = opt.mean
        self.std = opt.std

        transformations = [
            # Rescale an image so that minimum side is equal to max_size,
            # keeping the aspect ratio of the initial image.
            A.SmallestMaxSize(max_size=min(opt.h, opt.w)),
            A.RandomCrop(height=opt.h, width=opt.w) if opt.augment else A.CenterCrop(height=opt.h, width=opt.w),
        ]

        if opt.camera_noise:
            transformations.append(A.GaussNoise(p=0.8))

        if opt.augment:  # augment on smaller dimensions for performance
            transformations.extend([
                A.HueSaturationValue(hue_shift_limit=20, 
                    sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.ChannelShuffle(p=0.5),
                A.HorizontalFlip(p=0.2),
            ])

        transformations.append(A.Normalize(mean=opt.mean, std=opt.std))

        bbox_params = A.BboxParams(
            format="coco",  # coco format: x_min, y_min, width, height
            min_area=50,  # < x pixels => drop bbox
            min_visibility=0.5,  # < original visibility => drop bbox
        )

        self.transform_fn = A.Compose(transformations, bbox_params=bbox_params)

    def gen_map(self, shape, xy: np.ndarray, mask=None, sigma=2, cutoff=1e-3, 
            normalize=False, bleed=True):
        """
        Generates a single belief map of 'shape' for each point in 'xy'.

        Parameters
        ----------
        shape: tuple
            h x w of image
        xy: n x 2
            n points with x, y coordinates (image coordinate system)
        mask: n,
            zero-one mask to select points from xy
        sigma: scalar
            gaussian sigma
        cutoff: scalar
            set belief to zero if it is less then cutoff
        normalize: bool
            whether to multiply with the gaussian normalization factor or not

        Returns
        -------
        belief map: 1 x h x w  # num_classes x h x w
        """
        n = xy.shape[0]
        h, w = shape[:2] 

        if n == 0:
            return np.zeros((1, h, w), dtype=np.float32)

        if not bleed:
            wh = np.asarray([w - 1, h - 1])[None, :]
            mask_ = np.logical_or(xy[..., :2] < 0, xy[..., :2] > wh).any(-1)
            xy = xy.copy()
            xy[mask_] = np.nan

        # grid is 2 x h x h
        grid = np.array(np.meshgrid(np.arange(w), np.arange(h)), dtype=np.float32)
        # reshape grid to 1 x 2 x h x w
        grid = grid.reshape((1, 2, h, w))
        # reshape xy to n x 2 x 1 x 1
        xy = xy.reshape((n, 2, 1, 1))
        # compute squared distances to joints
        d = ((grid - xy) ** 2).sum(1)
        # compute gaussian
        b = np.nan_to_num(np.exp(-(d / (2.0 * sigma ** 2))))

        if normalize:
            b = b / np.sqrt(2 * np.pi) / sigma  # n x h x w

        # b is n x h x w
        b[(b < cutoff)] = 0

        if mask is not None:
            # set the invalid center point maps to all zero
            b *= mask[:, None, None]  # n x h x w

        b = b.max(0, keepdims=True)  # 1 x h x w

        # focal loss is different if targets aren't exactly 1
        # thus make sure that 1s are at discrete pixel positions 
        b[b >= 0.95] = 1
        return b  # 1 x h x w

    def item_transform(self, item):
        """
        Transform data for training.

        :param item: dictionary
            - image: h x w x 3
            - bboxes: n x 4; [[x, y, width, height], ...]
            - cids: n,

        :return: dictionary
            - image: 3 x 512 x 512
            - cpt_hm: 1 x 128 x 128 # num_classes x 128 x 128
            - cpt_off: n_max x 2 low resolution offset - [0, 1)
            - cpt_ind: n_max, low resolution indices - [0, 128^2)
            - cpt_mask: n_max,
            - wh: n_max x 2, low resolution width, height - [0, 128], [0, 128]
            - cls_id: n_max,
        """
        image = item["image"]  # h x w x 3
        bboxes = item['bboxes']  # n x 4
        cids = item["cids"]  # n,

        h, w = image.shape[:2]

        # prepare bboxes for transformation
        bbox_labels = np.arange(len(bboxes), dtype=np.float32)  # n,
        bboxes = np.append(bboxes, bbox_labels[:, None], axis=-1)

        transformed = self.transform_fn(image=image, bboxes=bboxes)
        image = np.array(transformed["image"], dtype=np.float32)
        image = image.transpose((2, 0, 1))  # 3 x h x w
        bboxes = np.array(transformed["bboxes"], dtype=np.float32).reshape(-1, 5)
        
        # bboxes can be dropped
        len_valid = len(bboxes)

        # to be batched we have to bring everything to the same shape
        cpt = np.zeros((self.n_max, 2), dtype=np.float32)
        # get center points of bboxes (image coordinates)
        cpt[:len_valid, 0] = bboxes[:, 0] + bboxes[:, 2] / 2  # x
        cpt[:len_valid, 1] = bboxes[:, 1] + bboxes[:, 3] / 2  # y

        cpt_mask = np.zeros((self.n_max,), dtype=np.uint8)
        cpt_mask[:len_valid] = 1

        # LOW RESOLUTION bboxes
        wh = np.zeros((self.n_max, 2), dtype=np.float32)
        wh[:len_valid, :] = bboxes[:, 2:-1] / self.down_ratio

        cls_id = np.zeros((self.n_max,), dtype=np.uint8)
        # the bbox labels help to reassign the correct classes
        cls_id[:len_valid] = cids[bboxes[:, -1].astype(np.int32)]

        # LOW RESOLUTION dimensions
        hl, wl = int(self.h / self.down_ratio), int(self.w / self.down_ratio)
        cpt = cpt / self.down_ratio

        # discrete center point coordinates
        cpt_int = cpt.astype(np.int32)

        cpt_ind = np.zeros((self.n_max,), dtype=np.int64)
        # index = y * wl + x
        cpt_ind[:len_valid] = cpt_int[:len_valid, 1] * wl + cpt_int[:len_valid, 0]

        cpt_off = np.zeros((self.n_max, 2), dtype=np.float32)
        cpt_off[:len_valid] = (cpt - cpt_int)[:len_valid]

        cpt_hms = []
        valid_cpt = cpt[cpt_mask.astype(np.bool)]  # n_valid x 2
        for i in range(self.num_classes):
            mask = (cls_id[:len_valid] == i)  # n_valid,
            xy = valid_cpt[mask]  # n x 2, valid entries for each class
            cpt_hms.append(self.gen_map((hl, wl), xy))  # each 1 x hl x wl
        
        cpt_hm = np.concatenate(cpt_hms, axis=0) 

        item = {
            "image": image,
            "cpt_hm": cpt_hm,
            "cpt_off": cpt_off,
            "cpt_ind": cpt_ind,
            "cpt_mask": cpt_mask,
            "wh": wh,
            "cls_id": cls_id,
        }

        return item
    

def item_filter(func, vis_thres):

    def inner(item):
        bboxes = item['bboxes']  # n x 4
        cids = item['cids']  # n,
        vis = item['visfracs']  # n,

        # visibility filtering
        mask = (vis > vis_thres)  # n,
        item["bboxes"] = bboxes[mask]  # n' x 4
        cids = cids[mask]  # n',

        # remap the class ids to our group assignment
        new_ids = np.array([CLSES_MAP[old_id] for old_id in cids], dtype=cids.dtype)
        item["cids"] = new_ids

        item = func(item)  # call the decorated function

        return item
    
    return inner


def main(opt):
    setattr(opt, "blend_path", "C:/Program Files/Blender Foundation/Blender 2.90")

    setattr(opt, "scene", )  # e.g. xy.blend -> xy
    setattr(opt, "blender_instances", 2)
    setattr(opt, "vis_thresh", 0.3)
    setattr(opt, "batch_size", 16)
    setattr(opt, "worker_instances", 4)
    setattr(opt, "weight_decay", 1 ** -3)

    setattr(opt, "num_samples", 2 * 10 ** 5)  # num. of samples to train for
    setattr(opt, "train_vis_interval", 200)
    setattr(opt, "val_vis_interval", 20)
    setattr(opt, "save_interval", 10 ** 4)
    setattr(opt, "val_interval", 10 ** 3)

    setattr(opt, "augment", True)
    setattr(opt, "camera_noise", False)  # camera noise augmentation

    transformation = Transformation(opt)
    item_transform = transformation.item_transform

    with ExitStack() as es:
        # Launch Blender instance. Upon exit of this script all Blender 
        # instances will be closed.
        bl = es.enter_context(
            btt.BlenderLauncher(
                scene=f"{opt.scene}.blend",
                script=f"{opt.scene}.blend.py",
                num_instances=opt.blender_instances,
                named_sockets=['DATA', 'CTRL'],
                blend_path=opt.blend_path,
            )
        )

        addr = bl.launch_info.addresses['DATA']
        # Setup a streaming dataset
        # Iterable datasets do not support shuffle
        ds = btt.RemoteIterableDataset(
            addr,
            max_items=10**8,
            item_transform=item_filter(item_transform, opt.vis_thres)
        )

        # Setup DataLoader
        dl = data.DataLoader(ds, batch_size=opt.batch_size, 
            num_workers=opt.worker_instances, shuffle=False)


        # Setup 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loss_fn = CenterLoss()
        model = get_model({"cpt_hm": opt.num_classes, "cpt_off": 2, "wh": 2})
        model.to(device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), 
            lr=opt.lr, weight_decay=opt.weight_decay)

        best_loss = 10 ** 8

        writer = SummaryWriter()  # save into ./runs folder
    
        train_meter = MetricMeter()
        val_meter = MetricMeter()

        # train for a given number of samples
        with tqdm(total=len(dl)) as pbar:
            for i, batch in enumerate(dl):  # iterate over batches
                i *= opt.batch_size

                batch = {k: v.to(device) for k, v in batch.items()}

                if i % opt.val_interval == 0 and i != 0:
                    model.eval()

                    output = model(batch["image"])
                    _, loss_dict = loss_fn(output, batch)

                    val_meter.update(loss_dict)
                    val_meter.to_writer(writer, "Val", n_iter=i)

                    if i % opt.val_vis_interval == 0  and i != 0:
                        batch = {k: v[:1].detach().clone().cpu() for k, v in batch.items()}
                        output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}
                        
                        add_dt(writer, "Val/DT", i, output, batch, opt)
                        add_gt(writer, "Val/GT", i, output, batch, opt)
                        add_hms(writer, "Val/HMS", i, output, batch)

                    if val_meter.get_avg("total_loss") <= best_loss:
                        best_loss = val_meter.get_avg("total_loss")

                        if isinstance(model, nn.DataParallel):
                            state_dict = model.module.state_dict()
                        else:
                            state_dict = model.state_dict()
                        
                        torch.save({
                            'loss': best_loss,
                            'nsample': i,
                            'model_state_dict': state_dict,
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, f"./models/best_model.pth")

                else:
                    model.train()

                    output = model(batch["image"])
                    loss, loss_dict = loss_fn(output, batch)

                    train_meter.update(loss_dict)
                    train_meter.to_writer(writer, "Train", n_iter=i)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if i % opt.train_vis_interval == 0 and i != 0:
                        batch = {k: v[:1].detach().clone().cpu() for k, v in batch.items()}
                        output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}
                        
                        add_dt(writer, "Train/DT", i, output, batch, opt)
                        add_gt(writer, "Train/GT", i, output, batch, opt)
                        add_hms(writer, "Train/HMS", i, output, batch)

                    if i % opt.save_interval == 0 and i != 0:
                        if isinstance(model, nn.DataParallel):
                            state_dict = model.module.state_dict()
                        else:
                            state_dict = model.state_dict()

                        torch.save({
                            'loss': best_loss,
                            'nsample': i,
                            'model_state_dict': state_dict,
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, f"./models/model_{i}.pth")    

                pbar.set_postfix(loss=loss.item())
                pbar.update()

                if i >= opt.num_samples:
                    break

            pbar.close()    


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)

    opt = Config("./configs/config.txt")
    print(opt)
    
    main(opt)