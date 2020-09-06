from contextlib import ExitStack

import albumentations as A
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from blendtorch import btt

from .utils import Config
from .model import get_model
from .decode import decode
from .visu import render


"""
RGB images -> data set: add padding, rescale to 512 x 512,
normalize by ImageNet mean and std -> data loader feed 
the model -> denormalize images, decode model output 
-> render the detections -> save detections
"""


class Transformation:

    def __init__(self, opt):
        self.h, self.w = opt.h, opt.w  # e.g. 512 x 512
        self.down_ratio = 4  # => low resolution 128 x 128
        # ImageNet stats
        self.mean = opt.mean
        self.std = opt.std

        transformations = [
            # Rescale an image so that maximum side is equal to max_size,
            # keeping the aspect ratio of the initial image.
            A.LongestMaxSize(max_size=max(opt.h, opt.w)),
            A.Normalize(mean=opt.mean, std=opt.std),
        ]

        bbox_params = A.BboxParams(
            format="coco",
            min_area=100,  # < 100 pixels => drop bbox
            min_visibility=0.2,  # < 20% of orig. vis. => drop bbox
        )

        self.transform_fn = A.Compose(transformations, bbox_params=bbox_params)

    def item_transform(self, item):
        """
        Transform data for inference.
        Ground truths can be used for AP calculations.

        :param item: dictionary
            - image: h x w x 3
            - bboxes: n x 4; [[x, y, width, height], ...]
            - cids: n,

        :return: dictionary
            - image: 3 x 512 x 512
            - gt (ground truth): 
                - bboxes: n x 4; [[x, y, width, height], ...]
                - cids: n,

        """
        image = item["image"]
        gt_image = image.copy()  # h x w x 3
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
        bw[x + bw > w] = w - x[x + bw > w]
        bh[y + bh > h] = h - y[y + bh > h]
        bboxes = np.concatenate((x, y, bw, bh), axis=-1)
        # note: further processing is done by albumentations, bboxes
        # are dropped when not satisfying min. area or visibility!

        # prepare bboxes for transformation
        bbox_labels = np.arange(len(bboxes), dtype=np.float32)
        bboxes = np.append(bboxes, bbox_labels[:, None], axis=-1)

        transformed = self.transform_fn(image=image, bboxes=bboxes)
        image = np.array(transformed["image"], dtype=np.float32)
        image = image.transpose((2, 0, 1))  # 3 x h x w
        pad = (0, right, 0, bottom) 

        bboxes = np.array(transformed["bboxes"], dtype=np.float32)
        cids = cids[bboxes[-1]]  # temporary labels for class id reassignment
        bboxes = bboxes[:, :-1]  # drop temporary labels
        # note: note some bboxes might be dropped, e.g. when to small
        # to be considered as valid!

        item = {
            "image": image,
            "gt": {
                "image": gt_image,
                "bboxes": bboxes,
                "cids": cids,
            },
        }
        return item


class ImageDataSet(data.Dataset):

    def __init__(self, root: str):
        super().__init__()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        return item


def main(opt):
    """
    HOW TO GET THE CORRECT BBOXES / CENTER POINTS 
    FROM THE 128 x 128 model output into the 
    arbitrary shaped original input image (before 512 x 512)   
    """
    transformation = Transformation(opt)
    item_transform = transformation.item_transform

    with ExitStack() as es:
        if not opt.replay:
            # Load images from data folder
            # TODO: create a dataset to infere on images 
            ds = ImageDataSet(root=opt.data_folder)
        else:
            # Otherwise we replay from file.
            name = os.path.basename(opt.scene)
            ds = btt.FileDataset(f'./data/record_{name}', item_transform=item_transform)
            shuffle = False

        # try to over fit on a single example
        ds = data.Subset(ds, indices=[0])

        # Setup DataLoader and iterate
        dl = data.DataLoader(ds, batch_size=opt.batch_size, num_workers=opt.worker_instances, shuffle=shuffle)

        # batch = next(iter(dl))
        # if opt.replay:
        #   print(batch["gt"]["bboxes"])
        #   print(batch["gt"]["cids"])
        # plt.imshow(batch["image"].squeeze(0).permute(1, 2, 0).numpy())
        # plt.show()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        PATH = "./models/center_net.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, PATH)

        checkpoint = torch.load(PATH)
        heads = {"cpt_hm": opt.num_classes, "cpt_off": 2, "wh": 2}
        model = get_model(heads)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)

    opt = Config("./configs/config.txt")
    print(opt)

    main(opt)


