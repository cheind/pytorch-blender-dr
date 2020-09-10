import albumentations as A
import numpy as np
from pathlib import Path
import json
import cv2
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from .utils import Config
from .model import get_model
from .decode import decode, filter_dets
from .visu import render


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

        # bbox_params = A.BboxParams(
        #     format="coco",
        #     min_area=100,  # < 100 pixels => drop bbox
        #     min_visibility=0.2,  # < 20% of orig. vis. => drop bbox
        # )

        self.transform_fn = A.Compose(transformations)  #, bbox_params=bbox_params)

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

            ground truth: 
            - bboxes: n x 4; [[x, y, width, height], ...]
            - cids: n,

        """
        image = item["image"]
        image_gt = image.copy()
        bboxes = item['bboxes']
        cids = item["cids"]

        h, w = image.shape[:2]

        # adjust bboxes to pass initial - check_bbox(bbox) - call
        # from the albumentations package
        # library can't deal with bbox corners outside of the image
        # x, y, bw, bh = np.split(bboxes, 4, -1)  # each n x 1
        # x[x < 0] = 0
        # y[y < 0] = 0
        # x[x > w] = w
        # y[y > h] = h
        # bw[x + bw > w] = w - x[x + bw > w]
        # bh[y + bh > h] = h - y[y + bh > h]
        # bboxes = np.concatenate((x, y, bw, bh), axis=-1)
        # note: further processing is done by albumentations, bboxes
        # are dropped when not satisfying min. area or visibility!

        # prepare bboxes for transformation
        # bbox_labels = np.arange(len(bboxes), dtype=np.float32)
        # bboxes = np.append(bboxes, bbox_labels[:, None], axis=-1)

        transformed = self.transform_fn(image=image)  #, bboxes=bboxes)
        image = np.array(transformed["image"], dtype=np.float32)
        h, w = image.shape[:2]  # rescaled image
        shape_pp = np.array(image.shape[:2], dtype=np.float32)

        right = self.w - w
        bottom = self.h - h

        image = cv2.copyMakeBorder(image, 0, bottom, 0, right, 
            cv2.BORDER_CONSTANT, value=0)
        image = image.transpose((2, 0, 1))  # 3 x h x w

        # bboxes = np.array(transformed["bboxes"], dtype=np.float32)
        # cids = cids[bboxes[:, -1].astype(np.int32)]  # temporary labels for class id reassignment
        # bboxes = bboxes[:, :-1]  # drop temporary labels
        # note: note some bboxes might be dropped, e.g. when to small
        # to be considered as valid!

        item.update({
            "image": image,  # 3 x h x w 
            "bboxes": bboxes,  # n x 4
            "cids": cids,  # n,
            "image_gt": image_gt,  # h x w x 3
            "shape_pp": shape_pp,  # pre padding shape
        })
        return item


class TLessRealDataset(data.Dataset):
    """Provides access to images, bboxes and classids for TLess real dataset.
    
    Download dataset
    https://bop.felk.cvut.cz/datasets/#T-LESS
    http://ptak.felk.cvut.cz/6DB/public/bop_datasets/tless_test_primesense_bop19.zip
    
    Format description 
    https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md
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
                rgbpath = scenepath / 'rgb' / f'{int(idx):06d}.png'
                assert rgbpath.exists()
                
                clsids = [int(e['obj_id']) for e in scene_gt[idx]]
                bboxes = [e['bbox_obj'] for e in scene_gt_info[idx]]
                
                self.all_rgbpaths.append(rgbpath)
                self.all_bboxes.append(np.array(bboxes))
                self.all_clsids.append(np.array(clsids))
                
    def __len__(self):
        return len(self.all_rgbpaths)
    
    def __getitem__(self, index):
        image = cv2.imread(str(self.all_rgbpaths[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # item: bundle of per image bboxes and class ids
        item = {
            "image": image,  # np.ndarray, h x w x 3
            "bboxes": self.all_bboxes[index],
            "cids": self.all_clsids[index],
            }
        item = self.item_transform(item)
        return item


def main(opt):
    transformation = Transformation(opt)
    item_transform = transformation.item_transform

    # Setup Dataset
    ds = TLessRealDataset(opt.real_path, item_transform)
    logging.info(f"Real data set size: {len(ds)}")

    # Setup DataLoader
    dl = data.DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(opt.model_path)
    logging.info(f"Loading model form: {opt.model_path}")
    heads = {"cpt_hm": opt.num_classes, "cpt_off": 2, "wh": 2}
    model = get_model(heads)
    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device=device)
    model.eval()
    logging.info(f"Loaded model from epoch: {epoch}")

    with torch.no_grad():
        for i, batch in enumerate(dl):
            batch = {k: v.to(device=device) for k, v in batch.items()}

            out = model(batch["image"])  # 1 x 3 x h x w
            dets = decode(out, opt.k)  # 1 x k x 6
            dets = filter_dets(dets, opt.thres)
            logging.info("Decoded model output!")

            image_gt = batch["image_gt"]  # 1 x h x w x 3, original image
            dets[..., :4] = dets[..., :4] * opt.down_ratio  # 512 x 512 space dets

            shape_pp = batch["shape_pp"]  # 1 x 2
            h_gt, w_gt = image_gt.size(1), image_gt.size(2)
            h_pp, w_pp = shape_pp[0]  # pre padded image height and width
            x_scale = w_gt / w_pp
            y_scale = h_gt / h_pp
            # scale bboxes to match original image space
            dets[..., 0] *= x_scale  # x
            dets[..., 1] *= y_scale  # y
            dets[..., 2] *= x_scale  # w
            dets[..., 3] *= y_scale  # h

            render(image_gt, dets, opt, show=False, save=True, 
                path=f"./data/{i:05d}.png", denormalize=False)

            if i > 100:
                break

    return  # exit


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)

    opt = Config("./configs/config.txt")
    print(opt)

    opt.mean = np.array(opt.mean, np.float32) 
    opt.std = np.array(opt.std, np.float32)

    main(opt)


