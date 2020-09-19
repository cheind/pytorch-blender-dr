import albumentations as A
import numpy as np
from pathlib import Path
import json
import cv2
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

# git clone https://github.com/cheind/pytorch-blender.git <DST>
# pip install -e <DST>/pkg_pytorch
from blendtorch import btt

from .utils import Config, FileStream
from .model import get_model
from .decode import decode, filter_dets
from .visu import render
from .main import CLSES_MAP, item_filter
from .evaluation import create_gt_anns, evaluate


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

        self.transform_fn = A.Compose(transformations)

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

        transformed = self.transform_fn(image=image)  #, bboxes=bboxes)
        image = np.array(transformed["image"], dtype=np.float32)
        h, w = image.shape[:2]  # rescaled image
        shape_pp = np.array(image.shape[:2], dtype=np.float32)

        right = self.w - w
        bottom = self.h - h

        image = cv2.copyMakeBorder(image, 0, bottom, 0, right, 
            cv2.BORDER_CONSTANT, value=0)
        image = image.transpose((2, 0, 1))  # 3 x h x w

        item.update({
            "image": image,  # 3 x h x w 
            "bboxes": bboxes,  # n x 4
            "cids": cids,  # n,
            "image_gt": image_gt,  # h x w x 3
            "shape_pp": shape_pp,  # pre padding shape
        })
        return item


class TLessTestDataset(data.Dataset):
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
        self.all_cids = []
        
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
                    if vis > opt.vis_thres:
                        filtered_bboxes.append(bbox)
                        filtered_cids.append(cid)

                if len(filtered_bboxes) > 0:  # only add non empty images
                    self.all_rgbpaths.append(rgbpath)
                    # list of n_objs x 4
                    self.all_bboxes.append(np.array(filtered_bboxes))
                    # list of n_objs,
                    self.all_cids.append(np.array(filtered_cids))
        
        # create image ids for evaluation, each image path has 
        # a unique id
        self.img_ids = list(range(len(self.all_rgbpaths)))

        # remap class ids to groups
        for i in range(len(self.all_cids)):
            # take the old id and map it to a new one
            new_ids = [CLSES_MAP[old_id] for old_id in self.all_cids[i]]
            self.all_cids[i] = np.array(new_ids, dtype=np.int32)

        # produce ground truth annotations
        rbg_relpaths = [p.relative_to(self.basepath) for p in self.all_rgbpaths]
        create_gt_anns(rbg_relpaths, self.all_bboxes, self.all_cids, "./evaluation/gt.json")
                
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
            "image_id": index,
            }
        # update the item dictionary
        item = self.item_transform(item)
        return item

    
def _to_float(x):
    """ Reduce precision to save memory as adviced by the COCO team. """
    return float(f"{x:.2f}")


def main(opt):
    transformation = Transformation(opt)
    item_transform = transformation.item_transform

    # Setup Dataset
    # if opt.replay:
    #     ds = btt.FileDataset(opt.record_path, item_transform=item_filter(item_transform, opt.vis_thres))
    #     # BUG: KeyError: 'image_id' klarerweise
    # else:
    ds = TLessTestDataset(opt.inference_path, item_transform)
    
    logging.info(f"Data set size: {len(ds)}")

    # Setup DataLoader
    dl = data.DataLoader(ds, batch_size=1, num_workers=opt.worker_instances, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Model
    logging.info("Loading model for inference...")
    checkpoint = torch.load(opt.model_path)

    heads = {"cpt_hm": opt.num_classes, "cpt_off": 2, "wh": 2}
    model = get_model(heads)

    epoch = checkpoint["epoch"]
    best_loss = checkpoint["loss"]
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device=device)
    model.eval()

    logging.info(f"Loaded model from: {opt.model_path}" \
                f" trained till epoch: {epoch} with best loss: {best_loss}")

    pred = []
    with torch.no_grad():
        for (i, batch) in tqdm(enumerate(dl), desc="Evaluation", total=len(dl)):
            batch = {k: v.to(device=device) for k, v in batch.items()}

            out = model(batch["image"])  # 1 x 3 x h x w
            dets = decode(out, opt.k)  # 1 x k x 6
            dets = filter_dets(dets, opt.model_thres)  # 1 x k' x 6

            image_gt = batch["image_gt"]  # 1 x h x w x 3, original image
            dets[..., :4] = dets[..., :4] * opt.down_ratio  # 512 x 512 space dets

            shape_pp = batch["shape_pp"]  # 1 x 2
            h_gt, w_gt = image_gt.size(1), image_gt.size(2)

            # Pre padded image height and width
            h_pp, w_pp = shape_pp[0]  
            x_scale = w_gt / w_pp
            y_scale = h_gt / h_pp

            # Scale bboxes to match original image space
            dets[..., 0] *= x_scale  # x
            dets[..., 1] *= y_scale  # y
            dets[..., 2] *= x_scale  # w
            dets[..., 3] *= y_scale  # h

            if opt.render_dets:
                render(image_gt, dets, opt, show=False, save=True, 
                    path=f"./data/{i:05d}.png", denormalize=False)

            # Create json results for AP evaluation
            image_id = int(batch["image_id"])
            dets = dets.squeeze(0).cpu().numpy()  # k' x 6

            bboxes = dets[..., :4]  # k' x 4
            scores = dets[..., 4]  # k',
            cids = dets[..., 5]  # k',

            for bbox, cid, score in zip(bboxes, cids, scores):
                pred.append({
                    "image_id": image_id,
                    "category_id": int(cid),
                    "bbox": list(map(_to_float, bbox)),
                    "score": _to_float(score),
                })
        
    # Save json results for evaluation
    json.dump(pred, open("./evaluation/pred.json", "w"))

    # Save console output
    with FileStream('./evaluation/AP.txt', parser=lambda x: x):
        evaluate("./evaluation/gt.json", "./evaluation/pred.json")
    
    logging.info("Finishd inference!")
    return  # exit


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)

    opt = Config("./configs/config.txt")
    print(opt)

    main(opt)


