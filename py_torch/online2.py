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
import copy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

# git clone https://github.com/cheind/pytorch-blender.git <DST>
# pip install -e <DST>/pkg_pytorch
from blendtorch import btt

from .train import train, eval
from .loss import CenterLoss
from .model import get_model
from .decode import decode, filter_dets
from .visu import render, COLORS
from .utils import Config, FileStream
from .evaluation import create_gt_anns, evaluate

from tqdm import tqdm
from .utils import MetricMeter
from .train import add_dt, add_gt, add_hms

from .coco_report import coco_eval, ap_values, draw_roc


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


class Trans:

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
            transformations.append(A.GaussNoise(p=1.0, var_limit=(10, 40)))

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

            added GT
            "bboxes": bboxes,  # n_max x 4
            "cids": cids,  # n_max,
        """

        # here we get with the item filter the class remapped and visibility filtered 
        # class ids and bboxes respectively
        image = item["image"]  # h x w x 3
        bboxes = item['bboxes']  # n x 4
        cids = item["cids"]  # n,

        #update_id = item["update_id"]

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

        # we can access here the ground truth cids and bboxes which are
        # pre filtered and valid after the augmentation transformation!
        bboxes_ = bboxes[:, :-1].copy()  # n x 4
        cids_ = cls_id[:len_valid].copy()  # n, 

        bboxes = np.zeros((self.n_max, 4), dtype=np.float32)
        cids = np.zeros((self.n_max,), dtype=np.int32)

        # to be stacked by the default collate_fn
        bboxes[:len_valid, :] = bboxes_
        cids[:len_valid] = cids_

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

            # ground truth to eval COCO on the fly
            "bboxes": bboxes,  # n_max x 4
            "cids": cids,  # n_max,

            # current update state of the remote data generator
            #"update_id": update_id,
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


def _to_float(x):
    """ Reduce precision to save memory as adviced by the COCO team. """
    return float(f"{x:.2f}")


def render_cls_distr(cls_distr, opt):
    """Create matplotlib figure with bar plot of class distribution.

    Args:
        cls_distr (dict): keys are labels 0 - 29
    """
    fig, ax = plt.subplots()

    # map class distribution from 0 - 29 to 1 - 30 to 0 - 5 (our labels)
    cd = [0 for _ in range(opt.num_classes)]  # initialize
    
    for old_cls_id in range(1, 31):  # labels 1 - 30
        new_cls_id = CLSES_MAP[old_cls_id]  # 0 - 5
        cd[new_cls_id] += cls_distr[old_cls_id - 1]  # 0 - 29
    
    rects = ax.bar(x=list(range(opt.num_classes)), 
        height=cd, color=COLORS[:opt.num_classes])

    total = sum(cd)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{_to_float(height)} / {int(height / total * 100)}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects)
    fig.tight_layout()
    fig.add_axes(ax)
    
    return fig


def add_cls_distr(writer, tag, n_iter, cls_distr, opt):
    """ Add class distribution bar plot to tensorboard writer """
    fig = render_cls_distr(cls_distr, opt)
    writer.add_figure(tag, fig, global_step=n_iter, close=True)


def add_pr_curve(writer, tag, n_iter, prec, cls_ids):
    fig, ax = plt.subplots()
    draw_roc(prec, cls_ids, ax)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    writer.add_figure(tag, fig, global_step=n_iter, close=True)


def main(opt):

    setattr(opt, "vis_thresh", 0.1)
    setattr(opt, "batch_size", 32)
    setattr(opt, "worker_instances", 4)
    #setattr(opt, "weight_decay", 10 ** -3)
    setattr(opt, "weight_decay", 0)

    setattr(opt, "num_samples", 2 * 10 ** 5)  # num. of samples to train for

    # all intervals must be multiple of batch size in current implementation!
    setattr(opt, "train_vis_interval", 16 * opt.batch_size)
    setattr(opt, "val_vis_interval", 2 * opt.batch_size)
    setattr(opt, "save_interval", 1024 * opt.batch_size)
    setattr(opt, "val_interval", 256 * opt.batch_size)
    setattr(opt, "val_len", 32 * opt.batch_size)
    # if total loss in TRAINING is below that we perform an update step
    # setattr(opt, "loss_thres", 5.0) TODO
    setattr(opt, "loss_thres", 0.01)

    setattr(opt, "lr", 0.001)  # times 8

    ### FOR DEBUGGING ####
    """
    setattr(opt, "num_samples", 40 * 16)  # num. of samples to train for
    setattr(opt, "train_vis_interval", 2 * 16)
    setattr(opt, "val_vis_interval", 1 * 16)
    setattr(opt, "save_interval", 8192)
    setattr(opt, "val_interval", 4 * 16)  
    setattr(opt, "val_len", 2 * 16)  
    setattr(opt, "loss_thres", 25.0)
    """
    ######################

    # decrease the threshold at 25, 50, 75% of total train samples used
    setattr(opt, "loss_thres_steps", [0.25, 0.5, 0.75])
    opt.loss_thres_steps = [opt.num_samples * step for step in opt.loss_thres_steps]
    opt.finished_loss_update = False
    opt.loss_thres_index = 0

    # the amount the loss threshold is decreased at each step
    setattr(opt, "loss_thres_decr", 1.3)

    #setattr(opt, "augment", True)
    setattr(opt, "augment", False)
    # setattr(opt, "camera_noise", True)  # camera noise augmentation
    setattr(opt, "camera_noise", False)  # camera noise augmentation

    setattr(opt, "eval_folder", "./runs/eval")

    setattr(opt, "launch_info", "/mnt/data/launch_info.json")
    # setattr(opt, "launch_info", "./launch_info.json")

    setattr(opt, "n_max", 25)  # max. num. of real objects per image

    setattr(opt, "k", 30)  # num. of extracted heat map peaks (>= n_max or generated objects per image)

    setattr(opt, "obj_step", 2)  # increase num_objects by this step for difficulty increase

    setattr(opt, "model_thres", 0.1)  # threshold for detection filtering

    # we start with uniform class distribution of the 1 to 30 class objects
    # in the T-Less data set and will later update the distribution to 
    # populate the scene with more objects we classified weakly
    cls_distr = {k:1 for k in range(30)}  # in blender intern labels: 1 - 30 are 0 - 29

    # start with a low number of objects first and increase the number in the validation 
    # step, is constraint: num_objects <= opt.n_max 
    # num_objects = 10 TODO
    num_objects = 12

    # over each validation run with lenght opt.val_len we start with
    # an image id of 0 and increase it, for each image we need a unique image
    # id to match the ground truth with the prediction in the COCO evaluation tool 
    image_id = 0

    # for each bbox annotation in the ground truth (typically multiple per image)
    # we need to have a unique annotation id (COCO format requieres this)
    gt_ann_id = 0

    # Note: both, image id and gt ann id, are reset after each validation run of opt.val_len
    
    print(opt)  # print options here since we adopt them from offline training

    trans = Trans(opt)

    with ExitStack() as es:
        # launch_info = btt.LaunchInfo.load_json(opt.launch_info)
        # data_addr = launch_info.addresses['DATA']
        # ctrl_addr = launch_info.addresses['CTRL']  # array of addresses

        # # Provides generic bidirectional communication with a single PyTorch instance
        # remotes = [btt.DuplexChannel(addr) for addr in ctrl_addr]

        # # initialize num. of objects and class distribution of remote
        # for remote in remotes:
        #     remote.send(num_objects=num_objects, object_cls_prob=cls_distr)

        """
        Other side (remote, blender):
        duplex = btb.DuplexChannel(btargs.btsockets['CTRL'], btargs.btid)
        
        We can receive a message (= a dictionary) like:
        msg = duplex.recv(timeoutms=0)
        Note: msg is None if timeout
        ...and thus we can alter some generation parameters in pre frame!

        And on this side (pytorch) we send a message like:
        remote.send(shape_params=subset.cpu().numpy(), shape_ids=subset_ids.numpy())
        
        Note: key word arguments get sent as dictionary by .send()
        """

        # Setup a streaming dataset
        # Iterable datasets do not support shuffle
        # ds = btt.RemoteIterableDataset(
        #     data_addr,
        #     max_items=10**8,
        #     item_transform=item_filter(trans.item_transform, opt.vis_thres)
        # )
        # shuffle = False

        opt.train_path = "/mnt/data/20201001_tless_refine/tless"
        ds = btt.FileDataset(opt.train_path, item_transform=item_filter(trans.item_transform, opt.vis_thres))
        shuffle = True
        #shuffle = False

        # Setup DataLoader
        dl = data.DataLoader(ds, batch_size=opt.batch_size, 
            num_workers=opt.worker_instances, shuffle=shuffle)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loss_fn = CenterLoss()
        model = get_model({"cpt_hm": opt.num_classes, "cpt_off": 2, "wh": 2}, pretrained=True)
        model.to(device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), 
            lr=opt.lr, weight_decay=opt.weight_decay)

        best_loss = 10 ** 8

        writer = SummaryWriter()  # save into ./runs folder

        # add the initial class distribution to tensorboard
        add_cls_distr(writer, "Class Distribution", 0, cls_distr, opt)

        num_tot_train_samples = 0

        num_val_samples = 0
        val_flag = False  # flag is True while validation samples are processed!

        val_meter = MetricMeter()
        train_meter = MetricMeter()
        
        # dictionary to build COCO json files for AP evaluation
        def empty_gt_dict():
            gt = {"images": [], "annotations": [], 
                  "categories": [
                      {"id": cls_id, "name": str(cls_id)} for cls_id in range(len(GROUPS)) 
                  ]}
            return gt
        
        # reset ground truths and predictions
        ground_truth = empty_gt_dict()
        prediction = []

        num_validations = int(opt.num_samples / opt.val_interval)
        val_samples = num_validations * opt.val_len
        # num. of targeted training samples plus validation sample = total samples used
        total_iterations = opt.num_samples + val_samples

        toggle_flag = True  # toggle between 2 different update strategies

        n = 0  # FOR DEBUGGING

        # train for a given number of samples
        with tqdm(total=total_iterations / opt.batch_size) as pbar:
            for i, batch in enumerate(dl):  # iterate over batches
                i *= opt.batch_size  # num of samples train + val

                batch = {k: v.to(device) for k, v in batch.items()}

                ### VISUALIZE STREAM DATA FOR DEBUGGING ###
                """
                for j in range(opt.batch_size):
                    batch_ = {k: v[j:j+1, ...] for k, v in batch.items()}

                    image = batch_["image"]  # original image

                    # build gt dets
                    inds = batch_["cpt_ind"]  # 1 x n_max
                    wh = batch_["wh"]  # 1 x n_max x 2
                    cids = batch_["cls_id"]  # 1 x n_max
                    mask = batch_["cpt_mask"].squeeze(0)  # n_max,
                    
                    ws = wh[..., 0]  # 1 x n_max
                    hs = wh[..., 1]  # 1 x n_max
                    wl = opt.w / opt.down_ratio
                    ys = torch.true_divide(inds, wl).int().float()  # 1 x n_max
                    xs = (inds % wl).int().float()  # 1 x n_max
                    scores = torch.ones_like(cids)  # 1 x n_max
                    
                    dets = torch.stack([xs - ws / 2, ys - hs / 2, 
                        ws, hs, scores, cids], dim=-1)  # 1 x n_max x 6

                    dets = dets[:, mask.bool()]  # 1 x n' x 6
                    dets[..., :4] = dets[..., :4] * opt.down_ratio

                    render(image, dets, opt, show=False, save=True, 
                        denormalize=True, path=f"./stream_images/{n:03d}.png", ret=False)
                    n += 1
                """
                ###

                # track update id
                # update_id = batch["update_id"]  # b,
                # update_id = update_id.float().mean().item()
                # writer.add_scalar("Update Id", update_id, global_step=i)

                # logging.debug(f"update_id: {update_id}")
                logging.debug(f"i: {i}")
                logging.debug(f"num_objects: {num_objects}")

                ### VALIDATION ###
                if i % opt.val_interval == 0 and i != 0 or val_flag:
                    logging.debug(f"VALIDATION -> val_flag: {val_flag}")

                    with torch.no_grad():
                        val_flag = True  # keep flag True till opt.val_len reached!

                        # update samples processed in current validation run
                        num_val_samples += opt.batch_size  

                        model.eval()

                        # model output dictionary, entry for each head's output
                        output = model(batch["image"])
                        _, loss_dict = loss_fn(output, batch)

                        val_meter.update(loss_dict)
                        val_meter.to_writer(writer, "Val", n_iter=i)

                        ### PREDICTION ###
                        gt_bboxes = batch["bboxes"].cpu().numpy()  # b x n_max x 4
                        gt_cids = batch["cids"].cpu().numpy()  # b x n_max
                        cpt_mask = batch["cpt_mask"].cpu().numpy()  # b x n_max

                        # b x k x 6; b x k x [bbox, score, cid]
                        dets_tensor = decode(output, opt.k)  # unfiltered  

                        is_crowd = 0  # no crowd annotations, individual objects labeled

                        for b in range(opt.batch_size):
                            # 1 x k' x 6 with possibly different k' each iteration
                            dets = filter_dets(dets_tensor[b, ...], opt.model_thres) 
                            dets[..., :4] = dets[..., :4] * opt.down_ratio  # 512 x 512
                            dets = dets.cpu().numpy()  # 1 x k' x 6

                            # store detections
                            k_filtered = dets.shape[1]  # = k'
                            for k in range(k_filtered):
                                prediction.append({
                                    "image_id": image_id,
                                    "category_id": int(dets[0, k, 5]),
                                    "bbox": list(map(_to_float, dets[0, k, :4])),
                                    "score": _to_float(dets[0, k, 4]),
                                })

                            # for each batch cpt_mask are all 1s and then all 0s
                            for idx in range(cpt_mask[b].sum()):
                                # area is used to group the AP evaluation into small, medium, large objects
                                # depending on the bbox area in pixels given fixed thresholds (see COCO)
                                area = gt_bboxes[b, idx, 2] * gt_bboxes[b, idx, 3]  # number of pixels

                                ground_truth["annotations"].append({
                                    "image_id": int(image_id),
                                    "category_id": int(gt_cids[b, idx]),
                                    "bbox": list(map(_to_float, gt_bboxes[b, idx, :])),
                                    "id": int(gt_ann_id),  # each annotation has to have a unique id
                                    "iscrowd": int(is_crowd),
                                    "area": int(area),
                                })

                                # increase, must be unique
                                gt_ann_id += 1

                            ground_truth["images"].append({
                                "id": int(image_id),
                            })

                            # increase, must be unique
                            image_id += 1

                        if i % opt.val_vis_interval == 0  and i != 0:
                            batch = {k: v[:1].detach().clone().cpu() for k, v in batch.items()}
                            output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}
                            
                            add_dt(writer, "Val/DT", i, output, batch, opt)
                            add_gt(writer, "Val/GT", i, output, batch, opt)
                            add_hms(writer, "Val/HMS", i, output, batch)

                        ### RESET ###
                        logging.debug(f"num_val_samples: {num_val_samples} >= opt.val_len: {opt.val_len} ?")
                        if num_val_samples >= opt.val_len:  
                            logging.debug(f"RESET -> num_val_samples: {num_val_samples}")

                            avg_total_loss_val = val_meter.get_avg("total_loss")
                            logging.debug(f"avg_total_loss_val: {avg_total_loss_val} <= best_loss: {best_loss} ?")
                            if avg_total_loss_val <= best_loss:
                                best_loss = val_meter.get_avg("total_loss")

                                if isinstance(model, nn.DataParallel):
                                    state_dict = model.module.state_dict()
                                else:
                                    state_dict = model.state_dict()
                                
                                torch.save({
                                    'loss': best_loss,
                                    'nsample': num_tot_train_samples,
                                    'model_state_dict': state_dict,
                                    'optimizer_state_dict': optimizer.state_dict(),
                                }, f"./models/best_model.pth")
                                logging.debug("Saved new best model!")


                            # reset validation counters/status trackers
                            val_flag = False
                            num_val_samples = 0

                            # we can reset the ids
                            image_id = 0
                            gt_ann_id = 0

                            ### AP EVALUATION ###
                            gtFile = f"{opt.eval_folder}/gt{num_tot_train_samples}.json"
                            dtFile = f"{opt.eval_folder}/dt{num_tot_train_samples}.json"

                            gt_ann_len = len(ground_truth["annotations"])
                            logging.debug(f"RESET -> ground_truth #annotations: {gt_ann_len}")
                            logging.debug(f"RESET -> #predictions: {len(prediction)}")

                            # we save the ground_truths and predictions after each validation run
                            json.dump(ground_truth, open(gtFile, "w"))
                            json.dump(prediction, open(dtFile, "w"))

                            # reset ground truths and predictions
                            ground_truth = empty_gt_dict()
                            prediction = []

                            # when under a certain training loss treshold we adapt the class distribution
                            # to favor the weakly classified samples
                            train_total_loss = train_meter.get_avg("total_loss")
                            logging.debug(f"train_total_loss: {train_total_loss} < opt.loss_thres: {opt.loss_thres} ?")
                            
                            ### UPDATE ###
                            if train_total_loss < opt.loss_thres:

                                if toggle_flag:  # increase num. of objects per image to increase difficulty
                                    logging.debug(f"UPDATE -> num_objects: {num_objects}")
                                    num_objects += opt.obj_step
                                    if num_objects > opt.n_max:
                                        num_objects = opt.n_max

                                    for remote in remotes:
                                        # change num. of objects
                                        remote.send(num_objects=num_objects)
                                    logging.debug(f"UPDATE -> num_objects: {num_objects}")

                                else:  # change class distribution to increase difficulty
                                    logging.debug(f"UPDATE -> cls_distr: {cls_distr}")

                                    cocoGt = COCO(gtFile)
                                    cocoDt = cocoGt.loadRes(dtFile)

                                    # get class based AP metrics
                                    prec_tensor = coco_eval(cocoGt, cocoDt)

                                    add_pr_curve(writer, "PR Curve", 
                                        update_id, prec_tensor, cocoGt.getCatIds())

                                    precs = np.zeros((len(GROUPS), ))

                                    for cls_id in range(len(GROUPS)):  # cls_id: 0 - 5 
                                        # precision over all IoUs 0.5 - 0.95 for a single class
                                        ap = ap_values(prec=prec_tensor,
                                            klass=cls_id)  # array: Precision(Recalls)
                                        ap = ap.mean(0)
                                        logging.debug(f"cls_id: {cls_id}, ap: {ap}")
                                        precs[cls_id] = ap

                                    weakest_cls = np.argmin(precs, axis=0)  # our labels 0 - 5
                                    logging.debug(f"UPDATE -> weakest_cls(0 - 5): {weakest_cls}")

                                    # GROUPS maps tless labels to our class groups e.g. 0: [1, 2, 3, 4],...
                                    weakest_tless_clses = GROUPS[weakest_cls]  # tless labels 1 - 30

                                    # blender intern 0 - 29
                                    weakest_tless_clses = [cls_id - 1 for cls_id in weakest_tless_clses]
                                    logging.debug(f"UPDATE -> weakest_tless_clses(0 - 29): {weakest_cls}")

                                    # add +1 over all members of the weakest class 
                                    n_weak_members = len(weakest_tless_clses)
                                    # thus add 1.0 / all weakest class members
                                    value = 1.0 / n_weak_members 

                                    # update the class distribution
                                    for cls_id in weakest_tless_clses:
                                        cls_distr[cls_id] += value

                                    # change class distribution s.t. the low AP classes are more likely
                                    for remote in remotes:
                                        # changed class distribution, cls ids blender intern 0 - 29
                                        remote.send(object_cls_prob=cls_distr)

                                    # add the new class distribution to tensorboard
                                    add_cls_distr(writer, "Class Distribution", update_id, 
                                        cls_distr, opt)
                                    
                                    logging.debug(f"UPDATE -> cls_distr: {cls_distr}")
                                
                                # alternate between changing num. of objects and class distribution
                                if num_objects == opt.n_max:
                                    # always change class distribution if max objs reached
                                    toggle_flag = False  
                                else:  
                                    toggle_flag = not toggle_flag 

                            logging.debug("Reset MetricMeters!")
                            # reset meters too
                            val_meter = MetricMeter()  # then avg is tracked over opt.val_len samples
                            # also build a new trainings progress tracker for the avg training loss
                            train_meter = MetricMeter()

                ### TRAINING ###
                else:
                    num_tot_train_samples += opt.batch_size  # update samples used for training
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

                    if (num_tot_train_samples % opt.save_interval == 0 
                        and num_tot_train_samples != 0):
                        if isinstance(model, nn.DataParallel):
                            state_dict = model.module.state_dict()
                        else:
                            state_dict = model.state_dict()

                        torch.save({
                            'loss': best_loss,
                            'nsample': num_tot_train_samples,
                            'model_state_dict': state_dict,
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, f"./models/model_{num_tot_train_samples}.pth")   

                    logging.debug(f'avg total loss train from meter: {train_meter.get_avg("total_loss")}')
                    logging.debug(f"opt.loss_thres: {opt.loss_thres}")    
                     
                # update the bar every batch
                pbar.set_postfix(loss=loss.item())
                pbar.update()

                if not opt.finished_loss_update: 
                    loss_thres_step = opt.loss_thres_steps[opt.loss_thres_index]
                    logging.debug(f"num_tot_train_samples: {num_tot_train_samples} >= loss_thres_step: {loss_thres_step} ?")
                    if num_tot_train_samples >= loss_thres_step:
                        logging.debug(f"Loss thres before update: {opt.loss_thres}")
                        opt.loss_thres -= opt.loss_thres_decr
                        opt.loss_thres_index += 1
                        if opt.loss_thres_index == len(opt.loss_thres_steps):
                            opt.finished_loss_update = True
                            logging.debug(f"Finished loss update: {opt.finished_loss_update}")

                ### EXIT ###
                if num_tot_train_samples >= opt.num_samples:
                    # if we exceed/meet the amount of training samples quit training!
                    logging.debug(f"Exit training at: {num_tot_train_samples} samples!")
                    break

            pbar.close()    


if __name__ == '__main__':
    import logging

    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.DEBUG)

    opt = Config("./configs/config.txt")
    
    main(opt)