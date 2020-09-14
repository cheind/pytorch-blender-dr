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

    def __init__(self, opt, remap_clses=False):
        self.h, self.w = opt.h, opt.w  # e.g. 512 x 512
        self.num_classes = opt.num_classes  # num. of object classes
        self.n_max = opt.n_max  # num. of max objects per image
        self.down_ratio = opt.down_ratio  # => low resolution 128 x 128
        # ImageNet stats
        self.mean = opt.mean
        self.std = opt.std
        self.remap_clses = remap_clses

        transformations = [
            A.RGBShift(p=0.5),
            A.RandomBrightness(limit=0.1, p=0.2),
            A.ChannelShuffle(p=0.5),
            A.HorizontalFlip(p=0.2),
            A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.2)
        ] if opt.augment and not opt.test else []

        transformations.extend([
            # Rescale an image so that minimum side is equal to max_size,
            # keeping the aspect ratio of the initial image.
            A.SmallestMaxSize(max_size=min(opt.h, opt.w)),
            A.RandomCrop(height=opt.h, width=opt.w) if opt.augment else A.CenterCrop(height=opt.h, width=opt.w),
            A.Normalize(mean=opt.mean, std=opt.std),
        ])

        bbox_params = A.BboxParams(
            format="coco",
            min_area=50,  # < 100 pixels => drop bbox
            min_visibility=0.5,  # < 20% of orig. vis. => drop bbox
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
        # pixels are rectangles => if floating point coordinate
        # isn't exactly on the pixel center => not exactly 1 at 
        # the discrete positions!! solution: broaden peak
        b[b >= 0.95] = 1
        return b

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
        image = item["image"]
        bboxes = item['bboxes']
        cids = item["cids"]

        h, w = image.shape[:2]

        # adjust bboxes to pass initial - check_bbox(bbox) - call
        # from the albumentations package
        # library can't deal with bbox corners outside of the image
        x, y, bw, bh = np.split(bboxes, 4, -1)  # each n x 1
        x[x < 0] = 0
        y[y < 0] = 0
        x[x > w] = w
        y[y > h] = h
        # in the phot realistic blender renders some width and hights are <0 !!
        # => clip values
        bw = np.clip(bw, 0, w)
        bh = np.clip(bh, 0, h)
        # bring bbox inside the image to be used with albumentations
        bw[x + bw > w] = w - x[x + bw > w]
        bh[y + bh > h] = h - y[y + bh > h]

        mask = np.logical_or(bw == 0, bh == 0).reshape(-1)  # n,
        bboxes = np.concatenate((x, y, bw, bh), axis=-1)  # n x 4
        bboxes = bboxes[~mask]
        # note: further processing is done by albumentations, bboxes
        # are dropped when not satisfying min. area or visibility!

        # prepare bboxes for transformation
        bbox_labels = np.arange(len(bboxes), dtype=np.float32)
        bboxes = np.append(bboxes, bbox_labels[:, None], axis=-1)


        # import matplotlib.pyplot as plt
        # try:
        #     transformed = self.transform_fn(image=image, bboxes=bboxes)
        # except Exception as e:
        #     img, bboxes, cids = item['image'], item['bboxes'], item['cids']
        #     H, W = img.shape[:2]  # img: h x w x 3
        #     DPI = 96
        #     fig = plt.figure(frameon=False, figsize=(W*2/DPI,H*2/DPI), dpi=DPI)
        #     axs = fig.add_axes([0,0,1.0,1.0])
        #     axs.imshow(img, origin='upper')
        #     for cid, bbox in zip(cids,bboxes):
        #         rect = patches.Rectangle(bbox[:2],bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')
        #         axs.add_patch(rect)
        #         axs.text(bbox[0]+10, bbox[1]+10, f'Class {cid.item()}', fontsize=18)
        #     axs.set_axis_off()
        #     axs.set_xlim(0,W-1)
        #     axs.set_ylim(H-1,0)

        #     plt.savefig("./debug/weird_bboxes.png")
        #     plt.close(fig)
        #     print(item["bboxes"])
            
        #     print(e); raise RuntimeError


        # img, bboxes, cids = item['image'], item['bboxes'], item['cids']
        # H, W = img.shape[:2]  # img: h x w x 3
        # DPI = 96
        # fig = plt.figure(frameon=False, figsize=(W*2/DPI,H*2/DPI), dpi=DPI)
        # axs = fig.add_axes([0,0,1.0,1.0])
        # axs.imshow(img, origin='upper')
        # for cid, bbox in zip(cids,bboxes):
        #     rect = patches.Rectangle(bbox[:2],bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')
        #     axs.add_patch(rect)
        #     axs.text(bbox[0]+10, bbox[1]+10, f'Class {cid.item()}', fontsize=18)
        # axs.set_axis_off()
        # axs.set_xlim(0,W-1)
        # axs.set_ylim(H-1,0)

        # plt.savefig("./debug/normal_bboxes.png")
        # plt.close(fig)


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

        if self.remap_clses:
            for i, cid in enumerate(cls_id[:len_valid]):
                # map class ids from e.g. 1 to 30 -> 0 to 5 
                cls_id[i] = CLSES_MAP[cid]

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


class TLessTrainDataset(data.Dataset):
    """
    Provides access to images, bboxes and classids 
    for TLess real dataset.
    """

    def __init__(self, basepath, item_transform=None):

        self.basepath = Path(basepath)
        self.item_transform = item_transform
        assert self.basepath.exists()
        
        # 000001, 000002,...
        self.all_rgbpaths = []
        self.all_bboxes = []
        self.all_clsids = []
        
        scenes = [f for f in self.basepath.iterdir() if f.is_dir()]
        for scenepath in scenes:
            with open(scenepath / 'scene_gt.json', 'r') as fp:
                scene_gt = json.loads(fp.read())
            with open(scenepath / 'scene_gt_info.json', 'r') as fp:
                scene_gt_info = json.loads(fp.read())
                
            for idx in scene_gt.keys():
                paths = [scenepath / 'rgb' / f'{int(idx):06d}.{ext}' for ext in ['png', 'jpg']]
                paths = [p for p in paths if p.exists()]
                assert len(paths)==1
                rgbpath = paths[0]
                
                clsids = [int(e['obj_id']) for e in scene_gt[idx]]
                bboxes = [e['bbox_obj'] for e in scene_gt_info[idx]]
                
                self.all_rgbpaths.append(rgbpath)
                # list of n_objs x 4
                self.all_bboxes.append(np.array(bboxes))
                self.all_clsids.append(np.array(clsids))
        
        # create image ids for evaluation, each image path has 
        # a unique id
        self.img_ids = list(range(len(self.all_rgbpaths)))

        # remap class ids to groups
        for i in range(len(self.all_clsids)):
            # take the old id and map it to a new one
            new_ids = [CLSES_MAP[old_id] for old_id in self.all_clsids[i]]
            self.all_clsids[i] = np.array(new_ids, dtype=np.int32)

        logging.warning("Remapped class ids here, don't remap in 'item_transform' !!")
                
    def __len__(self):
        return len(self.all_rgbpaths)
    
    def __getitem__(self, index: int):
        image = cv2.imread(str(self.all_rgbpaths[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # item: bundle of per image bboxes and class ids
        item = {
            "image": image,  # np.ndarray, h x w x 3
            "bboxes": self.all_bboxes[index],
            "cids": self.all_clsids[index],
        }
        
        # build new item dictionary
        item = self.item_transform(item)
        return item


def iterate(dl):
    DPI=96
    for step, item in enumerate(dl):
        img, bboxes, cids = item['image'], item['bboxes'], item['cids']
        H, W = img.shape[2:]  # img: b x 3 x h x w
        fig = plt.figure(frameon=False, figsize=(W*2/DPI,H*2/DPI), dpi=DPI)
        axs = [fig.add_axes([0,0,0.5,0.5]), fig.add_axes([0.5,0.0,0.5,0.5]), fig.add_axes([0.0,0.5,0.5,0.5]), fig.add_axes([0.5,0.5,0.5,0.5])]
        for i in range(img.shape[0]):
            axs[i].imshow(img[i].permute(1, 2, 0), origin='upper')
            for cid, bbox in zip(cids[i],bboxes[i]):
                rect = patches.Rectangle(bbox[:2],bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')
                axs[i].add_patch(rect)
                axs[i].text(bbox[0]+10, bbox[1]+10, f'Class {cid.item()}', fontsize=18)
            axs[i].set_axis_off()
            axs[i].set_xlim(0,W-1)
            axs[i].set_ylim(H-1,0)
        fig.savefig(f'./data/output_{step}.png')
        plt.close(fig)


def main(opt):
    # CARE REMAP CLSES ONLY ON PTB DATA
    transformation = Transformation(opt, remap_clses=False)
    item_transform = transformation.item_transform

    with ExitStack() as es:
        if not opt.replay:
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

            # Setup a streaming dataset
            ds = btt.RemoteIterableDataset(
                bl.launch_info.addresses['DATA'],
                item_transform=item_transform
            )
            # Iterable datasets do not support shuffle
            shuffle = False

            # Limit the total number of streamed elements
            ds.stream_length(4)

            # Setup raw recording if desired
            if opt.record:
                ds.enable_recording(opt.record_path)
        else:
            # Otherwise we replay from file.
            ds = btt.FileDataset(opt.record_path, item_transform=item_transform)
            shuffle = False
        
        ds = TLessTrainDataset(opt.real_train_path, item_transform)
        logging.info(f"Dataset: {ds.__class__.__name__}")

        logging.info(f"Num. of samples: {len(ds)}")
        logging.info(f"Batch size: {opt.batch_size}")

        # Setup Dataset: train, validation split
        split = int(0.9 * len(ds))
        logging.info(f"Split data set at sample nr. {split}")
        train_ds = data.Subset(ds, indices=list(range(split)))
        val_ds = data.Subset(ds, indices=list(range(split, len(ds))))

        logging.info(f"Training data size: {len(train_ds)}")
        logging.info(f"Validation data size: {len(val_ds)}")

        # Setup DataLoader: train, validation split
        train_dl = data.DataLoader(train_ds, batch_size=opt.batch_size, 
            num_workers=opt.worker_instances, shuffle=True)
        val_dl = data.DataLoader(val_ds, batch_size=opt.batch_size, 
            num_workers=opt.worker_instances, shuffle=False)

        if opt.record:
            print("Generating images of the recorded data...")
            dl = data.DataLoader(ds, batch_size=4, num_workers=opt.worker_instances, shuffle=False)
            iterate(dl)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"On device: {device}")

        # heads - head_name: num. channels of model
        heads = {"cpt_hm": opt.num_classes, "cpt_off": 2, "wh": 2}

        loss_fn = CenterLoss()

        if opt.test:  # do inference
            logging.info(f"Loading model for inference...")
            model = get_model(heads)

            checkpoint = torch.load(opt.model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint["epoch"]
            best_loss = checkpoint["loss"]

            logging.info(f"Loaded model from: {opt.model_path}" \
                f" and trained till epoch: {epoch}")

            model.to(device=device)
            model.eval()

            # batch size of 1
            val_dl = data.DataLoader(val_ds, batch_size=1, 
                num_workers=opt.worker_instances, shuffle=False)

            for i, batch in enumerate(val_dl): 
                # batch = next(iter(val_dl))  
                batch = {k: v.to(device=device) for k, v in batch.items()}

                # check if output hm are working correct!
                # import matplotlib.pyplot as plt
                # image = batch["image"].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                # fig = plt.figure()
                # axs = fig.add_axes([0,0,1.0,1.0])
                # axs.imshow(image, origin="upper")
                # bboxes = batch["bboxes"].squeeze(0).detach().cpu().numpy()
                # cids = batch["cids"].squeeze(0).detach().cpu().numpy()
                # print(cids)
                # print(transformation.map_cls)
                # for cid, bbox in zip(cids, bboxes):
                #     rect = patches.Rectangle(bbox[:2],bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')
                #     axs.add_patch(rect)
                #     axs.text(bbox[0]+10, bbox[1]+10, str(cid.item()), fontsize=18)
                # j = 0
                # plt.savefig(f"./debug/multi_{j}.png"); j += 1
                # for hm in batch["cpt_hm"].squeeze(0):
                #     hm = hm.detach().cpu().numpy()
                #     plt.imshow(hm, origin="upper", cmap="Greys", vmin=0, vmax=1)
                #     plt.savefig(f"./debug/multi_{j}.png"); j += 1
                # return

                with torch.no_grad():
                    out = model(batch["image"])
                    loss, loss_dict = loss_fn(out, batch)
                    print(loss_dict)

                    dets = decode(out, opt.k)  # 1 x k x 6
                    dets = filter_dets(dets, opt.thres)

                    image = batch["image"]
                    dets[..., :4] = dets[..., :4] * opt.down_ratio
                    render(image, dets, opt, show=False, save=True, 
                        path=f"./data/{i:05d}pred.png")

                    # # render ground truths
                    # wh = batch["wh"]  # 1 x n_max x 2
                    # ind = batch["cpt_ind"]  # 1 x n_max
                    # off = batch["cpt_off"]  # 1 x n_max x 2
                    # mask = batch["cpt_mask"].bool()  # 1 x n_max

                    # wh = wh[mask].view(1, -1, 2)  # 1 x n x 2
                    # off = off[mask].view(1, -1, 2)  # 1 x n x 2
                    # ind = ind[mask].view(1, -1)  # 1 x n
                    # empty = torch.zeros(1, 2, 128, 128, dtype=torch.float32).to(device=device)
                    # # generate output from ground truth
                    # empty = empty.view(1, 2, -1)  # 1 x num_classes x 128**2
                    # ind = ind.unsqueeze(1).expand(-1, 2, -1)  # 1 x 2 x n
                    # wh = wh.permute(0, 2, 1)  # 1 x 2 x n
                    # off = off.permute(0, 2, 1)  # 1 x 2 x n
                    # wh = empty.scatter(dim=-1, index=ind, src=wh).view(1, 2, 128, 128)
                    # off = empty.scatter(dim=-1, index=ind, src=off).view(1, 2 ,128, 128)
                    
                    # import matplotlib.pyplot as plt
                    # image_ = batch["cpt_hm"].squeeze(0).cpu()
                    # fig = plt.figure()
                    # axs = fig.add_axes([0,0,1.0,1.0])
                    # image_ = image_.max(0, keepdims=False)[0]
                    # axs.imshow(image_, origin="upper")
                    # plt.savefig("./data/cpt_hm.png")

                    # axs.imshow(wh[0, :, ...].cpu().max(0, keepdims=False)[0], 
                    #     origin="upper")
                    # plt.savefig("./data/wh.png")

                    # axs.imshow(off[0, :, ...].cpu().max(0, keepdims=False)[0], 
                    #     origin="upper")
                    # plt.savefig("./data/off.png")

                    # out = {"cpt_hm": batch["cpt_hm"],
                    #        "cpt_off": off, "wh": wh}
                    # dets = decode(out, opt.k)  # 1 x k x 6
                    # dets = filter_dets(dets, opt.thres)
                    # dets[..., :4] = dets[..., :4] * opt.down_ratio
                    # render(image, dets, opt, show=False, save=True, 
                    #     path=f"./data/{i:05d}gt.png")
                break

            return  # exit

        if opt.resume:  # resume training
            model = get_model(heads)
            model.to(device=device)
            optimizer = optim.Adam(model.parameters(), 
                lr=opt.lr, weight_decay=opt.weight_decay)
            checkpoint = torch.load(opt.model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint["loss"]
            start_epoch = checkpoint["epoch"] + 1
        else:  # train from scratch
            model = get_model(heads)
            model.to(device=device)
            optimizer = optim.Adam(model.parameters(), 
                lr=opt.lr, weight_decay=opt.weight_decay)
            best_loss = 10 ** 8
            start_epoch = 1

        writer = SummaryWriter()  # save into ./runs folder

        # tensorboard --logdir=runs --bind_all
        # bind_all -> when training on e.g. gpu server

        logging.info("Entering trainings loop...")

        for epoch in range(start_epoch, opt.num_epochs + start_epoch):
            meter = train(epoch, model, optimizer, train_dl, 
                device, loss_fn, writer, opt)
            
            total_loss = meter.get_avg("total_loss")
            logging.info(f"Loss: {total_loss} at epoch: {epoch} / {start_epoch + opt.num_epochs}")

            torch.save({
                'loss': best_loss,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "./models/model_last.pth")

            if epoch % opt.save_interval == 0:
                torch.save({
                    'loss': best_loss,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"./models/model_{epoch}.pth")

            if epoch % opt.val_interval == 0:
                meter = eval(epoch, model, val_dl, device, loss_fn, writer, opt)

                if meter.get_avg("total_loss") <= best_loss:
                    best_loss = meter.get_avg("total_loss")
                    torch.save({
                        'loss': best_loss,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f"./models/best_model.pth")

        logging.info("Done training!")
        return  # exit


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)

    opt = Config("./configs/config.txt")
    print(opt)

    opt.mean = np.array(opt.mean, np.float32) 
    opt.std = np.array(opt.std, np.float32)

    main(opt)
