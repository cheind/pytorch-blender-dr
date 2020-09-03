from contextlib import ExitStack
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
import numpy as np
import os

import torch
from torch import optim
from torch.utils import data

from blendtorch import btt

from .utils import Config
from .train import train, eval
from .loss import CenterLoss
from .model import get_model


class Transformation:

    def __init__(self, opt):
        self.h, self.w = opt.h, opt.w  # e.g. 512 x 512
        self.num_classes = opt.num_classes  # num. of object classes
        self.n_max = opt.n_max  # num. of max objects per image
        self.down_ratio = 4  # => low resolution 128 x 128
        # ImageNet stats
        self.mean = opt.mean
        self.std = opt.std

        transformations = [
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
            min_area=100,  # < 100 pixels => drop bbox
            min_visibility=0.2,  # < 20% of orig. vis. => drop bbox
        )

        self.transform_fn = A.Compose(transformations, bbox_params=bbox_params)

    def item_transform(self, item):
        """
        Transform data for training.

        :param item: dictionary
            - image: h x w x 3
            - bboxes: n x 4; [[x, y, width, height], ...]
            - cids: n,

        :return: dictionary
            - image: 3 x 512 x 512
            - cpt_hm: num_classes x 128 x 128
            - cpt_off: n_max x 2 low resolution offset - [0, 1)
            - cpt_ind: n_max, low resolution indices - [0, 128^2)
            - cpt_mask: n_max,
            - wh: n_max, 2
            - cls_id: n_max,
        """
        image = item["image"]
        bboxes = item['bboxes']
        cids = item["cids"]

        # prepare bboxes for transformation
        bbox_labels = np.array(list(range(len(bboxes))), dtype=np.float32)
        bboxes = np.concatenate((bboxes, bbox_labels), axis=-1)

        transformed = self.transform_fn(image=image, bboxes=bboxes)
        image = np.array(transformed["image"], dtype=np.flaot32)
        image = image.transpose((2, 0, 1))  # 3 x h x w
        bboxes = np.array(transformed["bboxes"], dtype=np.float32)

        # bboxes can be dropped
        len_valid = len(bboxes)

        # to be batched we have to bring everything to the same shape
        cpt = np.zeros((self.n_max, 2), dtype=np.float32)
        # get center points of bboxes (image coordinates)
        cpt[:len_valid, 0] = bboxes[:, 0] + bboxes[:, 2] / 2
        cpt[:len_valid, 1] = bboxes[:, 1] + bboxes[:, 3] / 2

        cpt_mask = np.zeros((self.n_max, 2), dtype=np.uint8)
        cpt_mask[:len_valid] = 1

        wh = np.zeros((self.n_max, 2), dtype=np.float32)
        wh[:len_valid, :] = bboxes[:, 2:-1]

        cls_id = np.zeros((self.n_max,), dtype=np.uint8)
        # the bbox labels help to reassign the correct classes
        cls_id[:len_valid] = cids[bboxes[:, -1]]

        # low resolution dimensions
        hl, wl = self.h / self.down_ratio, self.w / self.down_ratio
        cpt = cpt / self.down_ratio

        # discrete center point coordinates
        cpt_int = cpt.astype(np.int32)

        cpt_ind = np.zeros((self.n_max,), dtype=np.int64)
        cpt_ind[:len_valid] = cpt_int[:, 1] * wl + cpt_int[:, ]

        cpt_off = np.zeros((self.n_max, 2), dtype=np.float32)
        cpt_off

        cpt_hm = gen_map((hl, wl), cpt, mask=cpt_mask)

        item = {
            "image": image,
            "cpt_hm": cpt_hm,
            "cpt_off": cpt_off,
            "cpt_ind": cpt_ind,
            "cpt_mask": cpt_mask,
            "wh": wh,
            "cls_id": cls_id
        }
        return item

    def post_process(self):
        pass  # denormalize


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
    transformation = Transformation(opt)
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
                name = os.path.basename(opt.scene)
                ds.enable_recording(f'./data/record_{name}')
        else:
            # Otherwise we replay from file.
            name = os.path.basename(opt.scene)
            ds = btt.FileDataset(f'./data/record_{name}', item_transform=item_transform)
            shuffle = True

        # Setup DataLoader and iterate
        dl = data.DataLoader(ds, batch_size=opt.batch_size, num_workers=opt.worker_instances, shuffle=shuffle)

        # Generate images of the recorded data
        if opt.record:
            iterate(dl)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #"""
        # heads - head_name: num. channels of model
        heads = {"cpt_hm": opt.num_classes, "cpt_off": 2, "wh": 2}
        model = get_model(heads)

        optimizer = optim.Adam(model.parameters(), opt.lr)

        loss_fn = CenterLoss()

        for epoch in opt.num_epochs:
            epoch += 1

            train(epoch, model, optimizer, dl, device, loss_fn)

            if epoch % opt.val_interval == 0:
                eval(epoch, model, dl, device, loss_fn)
        #"""



if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)

    opt = Config("./configs/config.txt")
    print(opt)

    main(opt)
