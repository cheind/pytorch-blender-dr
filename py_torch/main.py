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

    def __init__(self, opt):
        self.h, self.w = opt.h, opt.w  # e.g. 512 x 512
        self.num_classes = opt.num_classes  # num. of object classes
        self.n_max = opt.n_max  # num. of max objects per image
        self.down_ratio = opt.down_ratio  # => low resolution 128 x 128
        # ImageNet stats
        self.mean = opt.mean
        self.std = opt.std

        transformations = [
            A.HueSaturationValue(hue_shift_limit=20, 
                sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.ChannelShuffle(p=0.5),
            A.HorizontalFlip(p=0.2),
        ] if opt.augment else []

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
        self.all_cids = []
        
        scenes = [f for f in self.basepath.iterdir() if f.is_dir()]
        for scenepath in scenes:
            is_bop_format = (scenepath / 'scene_gt.json').exists()
            is_old_format = (scenepath / 'gt.yml').exists()
            assert is_bop_format or is_old_format, 'Cannot determine format.'
            
            if is_bop_format:
                rgbpaths, bboxes, clsids = self._parse_bop_scene(scenepath, opt.vis_thres)
            else:
                rgbpaths, bboxes, clsids = self._parse_old_format(scenepath, opt.vis_thres)
                
            self.all_rgbpaths.extend(rgbpaths)
            self.all_bboxes.extend(bboxes)
            self.all_cids.extend(clsids)
        
        # create image ids for evaluation, each image path has 
        # a unique id
        self.img_ids = list(range(len(self.all_rgbpaths)))

        # remap class ids to groups
        for i in range(len(self.all_cids)):
            # take the old id and map it to a new one
            new_ids = [CLSES_MAP[old_id] for old_id in self.all_cids[i]]
            self.all_cids[i] = np.array(new_ids, dtype=np.int32)
                
    def __len__(self):
        return len(self.all_rgbpaths)
    
    def __getitem__(self, index: int):
        image = cv2.imread(str(self.all_rgbpaths[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # item: bundle of per image bboxes and class ids
        item = {
            "image": image,  # np.ndarray, h x w x 3
            "bboxes": self.all_bboxes[index],
            "cids": self.all_cids[index],
        }
        
        # build new item dictionary
        item = self.item_transform(item)

        return item

    def _parse_bop_scene(self, scenepath, vis_threshold):
        with open(scenepath / 'scene_gt.json', 'r') as fp:
            scene_gt = json.loads(fp.read())
        with open(scenepath / 'scene_gt_info.json', 'r') as fp:
            scene_gt_info = json.loads(fp.read())
            
        all_rgbpaths = []
        all_bboxes = []
        all_clsids = []
        
        for idx in scene_gt.keys():
            paths = [scenepath / 'rgb' / f'{int(idx):06d}.{ext}' for ext in ['png', 'jpg']]
            paths = [p for p in paths if p.exists()]
            assert len(paths)==1
            rgbpath = paths[0]

            clsids = [int(e['obj_id']) for e in scene_gt[idx]]
            bboxes = [e['bbox_obj'] for e in scene_gt_info[idx]]
            
            cids = [int(e['obj_id']) for e in scene_gt[idx]]
            bboxes = [e['bbox_visib'] for e in scene_gt_info[idx]]

            # visib_fract takes the inter-object coverage into account
            # thus we can avoid bboxes of covered objects
            # AND by taking the bbox_visib ones we have all bbox edges 
            # INSIDE of the image and hence no further preprocessing needed
            # before feeding into albumentation's transformations 
            visib_fracts = [e['visib_fract'] for e in scene_gt_info[idx]]

            filtered_bboxes, filtered_cids = [], []

            # filter bboxes by visibility:
            for bbox, cid, vis in zip(bboxes, cids, visib_fracts):
                if vis > vis_threshold:
                    filtered_bboxes.append(bbox)
                    filtered_cids.append(cid)

            if len(filtered_bboxes) > 0:  # only add non empty images                
                all_rgbpaths.append(rgbpath)
                # list of n_objs x 4
                all_bboxes.append(np.array(filtered_bboxes))
                # list of n_objs,
                all_clsids.append(np.array(filtered_cids))
            
        return all_rgbpaths, all_bboxes, all_clsids
    
    def _parse_old_format(self, scenepath, vis_threshold):
        del vis_threshold
        import yaml
        
        with open(scenepath / 'gt.yml', 'r') as fp:            
            scene_gt = yaml.load(fp.read(), Loader=yaml.Loader)
            
        all_rgbpaths = []
        all_bboxes = []
        all_clsids = []
        
        for idx in scene_gt.keys():
            rgbpath = scenepath / 'rgb' / f'{int(idx):04d}.png'
            #assert rgbpath.exists()
            
            clsids = [int(e['obj_id']) for e in scene_gt[idx]]
            bboxes = [e['obj_bb'] for e in scene_gt[idx]]
                       
            all_rgbpaths.append(rgbpath)
            all_bboxes.append(np.array(bboxes))
            all_clsids.append(np.array(clsids))
            
        return all_rgbpaths, all_bboxes, all_clsids            



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

            # Setup a streaming dataset
            ds = btt.RemoteIterableDataset(
                bl.launch_info.addresses['DATA'],
                item_transform=item_filter(item_transform, opt.vis_thres)
            )
            # Iterable datasets do not support shuffle
            shuffle = False

            # Limit the total number of streamed elements
            ds.stream_length(4)

            # Setup raw recording if desired
            if opt.record:
                ds.enable_recording(opt.record_path)
        elif opt.replay:
            # Otherwise we replay from file.
            ds = btt.FileDataset(opt.train_path, item_transform=item_filter(item_transform, opt.vis_thres))
            shuffle = True
        else:
            # if we stream or replay we need to apply a item filter on the job, whilst
            # the tless data set is static and thus item filtering (visibility) is done 
            # beforehand!
            ds = TLessTrainDataset(opt.train_path, item_transform)
            shuffle = True

        # Print Dataset stats
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
            num_workers=opt.worker_instances, shuffle=shuffle)
        val_dl = data.DataLoader(val_ds, batch_size=opt.batch_size, 
            num_workers=opt.worker_instances, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"On device: {device}")

        # heads - head_name: num. channels of model
        heads = {"cpt_hm": opt.num_classes, "cpt_off": 2, "wh": 2}

        loss_fn = CenterLoss()

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

        logging.info("Entering trainings loop, open tensorboard to view progress...")

        logging.info("Attach configuration file content to current tensorboard run...")
        writer.add_text("Configuration", str(opt), global_step=0)

        try:
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
        except KeyboardInterrupt:
            print("\n")
            logging.info("Manually stopped training!")
        finally:
            logging.info("Attach hyperparameter settings to current tensorboard run...")
            metrics = {"hparams/best_loss": best_loss}  # best loss on validation set
            writer.add_hparams(opt.hyperparams, metrics)

        return  # exit


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)

    opt = Config("./configs/config.txt")
    print(opt)

    # Relevant stats for tuning the training
    hyperparams =  {
        "batch_size": opt.batch_size,
        "lr": opt.lr,
        "weight_decay": opt.weight_decay,
        "n_max": opt.n_max,  # how many objects are at most possible as GT objects?
        "augment": opt.augment,  # augmented or not?
        "train_path": opt.train_path,  # which data was used?
    }

    """ BUG in tensorboard? -> doesnt show string or ALL float/int values?? AND a new run is made even if my key is UNIQUE DOCS:
        add_hparams(self, hparam_dict, metric_dict)
    |      Add a set of hyperparameters to be compared in TensorBoard.
    |
    |      Args:
    |          hparam_dict (dict): Each key-value pair in the dictionary is the
    |            name of the hyper parameter and it's corresponding value.
    |            The type of the value can be one of `bool`, `string`, `float`,
    |            `int`, or `None`.
    |          metric_dict (dict): Each key-value pair in the dictionary is the
    |            name of the metric and it's corresponding value. Note that the key used
    |            here should be unique in the tensorboard record. Otherwise the value
    |            you added by ``add_scalar`` will be displayed in hparam plugin. In most
    |            cases, this is unwanted.
    |
    |      Examples::
    |
    |          from torch.utils.tensorboard import SummaryWriter
    |          with SummaryWriter() as w:
    |              for i in range(5):
    |                  w.add_hparams({'lr': 0.1*i, 'bsize': i},
    |                                {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})
    |
    |      Expected result:
    |
    |      .. image:: _static/img/tensorboard/add_hparam.png
    |         :scale: 50 %
    """

    setattr(opt, "hyperparams", hyperparams)

    main(opt)
