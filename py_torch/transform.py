import albumentations as A
import numpy as np 
from typing import List
import cv2

from .utils import generate_heatmap
from .constants import MAPPING

class Transform:

    def __init__(self, opt, bbox_format='coco',
        augmentations: List = None, normalize=True,
        resize_crop=False, bboxes=True, filter=False):
        super().__init__()
        self.h, self.w = opt.h, opt.w  # e.g. 512 x 512
        self.num_classes = opt.num_classes  # num. of object classes
        self.n_max = opt.n_max  # num. of max objects per image
        self.down_ratio = opt.down_ratio  # => low resolution 128 x 128
        self.mean = opt.mean
        self.std = opt.std

        self.train = opt.train 

        self.resize_crop = resize_crop
        self.normalize = normalize
        self.augmentations = augmentations
        self.bbox_format = bbox_format 
        self.bboxes = bboxes
        self.filter = filter

        if opt.train:
            if augmentations is None:
                # standard augmentation if not given explicitly
                transformations = [
                    A.HueSaturationValue(p=0.5),
                    A.ChannelShuffle(p=0.4),
                    A.HorizontalFlip(p=0.2),
                    A.GaussNoise(p=0.3),
                ]
            else:
                transformations = augmentations

            if resize_crop:
                transformations.extend([
                    # Rescale an image so that minimum side is equal to max_size,
                    # keeping the aspect ratio of the initial image.
                    A.SmallestMaxSize(max_size=min(opt.h, opt.w)),
                    # A.CenterCrop(height=opt.h, width=opt.w)
                    A.RandomCrop(height=opt.h, width=opt.w),
                ])  
            
            if bboxes:
                bbox_params = A.BboxParams(
                    # pascal_voc, albumentatons, coco, yolo
                    format=bbox_format,  
                    # < min_area (in pixels) will drop bbox
                    min_area=50,  
                    # < min_visibility (relative to input bbox) will drop bbox
                    min_visibility=0.5,  
                )
            else:
                bbox_params = None
        else:
            transformations = [
                # Rescale an image so that maximum side is equal to max_size,
                # keeping the aspect ratio of the initial image.
                A.LongestMaxSize(max_size=max(opt.h, opt.w)),
                A.Normalize(mean=opt.mean, std=opt.std),
            ]
            bbox_params = None

        if normalize:  # zero mean, unit variance
            transformations.append(A.Normalize(mean=opt.mean, std=opt.std))

        self.transform_fn = A.Compose(transformations, 
            bbox_params=bbox_params)

        if opt.train:
            self.item_transform = self.item_transform_train
        else:
            self.item_transform = self.item_transform_test

        if filter:
            self.item_transform = self.item_filter(self.item_transform,
                opt.bbox_visibility_threshold)

    def item_filter(self, func, bbox_visibility_threshold):

        def inner(item):
            bboxes = item['bboxes']  # n x 4
            cids = item['cids']  # n,
            vis = item['visfracs']  # n,

            # visibility filtering
            mask = (vis > bbox_visibility_threshold)  # n,
            item["bboxes"] = bboxes[mask]  # n' x 4
            cids = cids[mask]  # n',

            # remap the class ids to our group assignment
            new_ids = np.array([MAPPING[old_id] for old_id in cids], dtype=cids.dtype)
            item["cids"] = new_ids
            item = func(item)  # call the decorated function
            return item
        
        return inner

    def item_transform_train(self, item: dict):
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
        """
        image = item["image"]  # h x w x 3
        bboxes = item['bboxes']  # n x 4
        cids = item["cids"]  # n,
        h, w = image.shape[:2]

        bbox_labels = np.arange(len(bboxes), dtype=np.float32)  # n,
        bboxes = np.append(bboxes, bbox_labels[:, None], axis=-1)
        transformed = self.transform_fn(image=image, bboxes=bboxes)
        image = np.array(transformed["image"], dtype=np.float32)
        image = image.transpose((2, 0, 1))  # 3 x h x w
        bboxes = np.array(transformed["bboxes"], dtype=np.float32).reshape(-1, 5)
        
        len_valid = len(bboxes)  # bboxes can be dropped

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
            cpt_hms.append(generate_heatmap((hl, wl), xy))  # each 1 x hl x wl
        
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

    def item_transform_test(self, item: dict):
        """
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

        transformed = self.transform_fn(image=image)
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

    def __call__(self, item):
        return self.item_transform(item)

def convert_bbox_format(bboxes, source_fmt, target_fmt, h, w):
    # pascal_voc: [x_min, y_min, x_max, y_max]
    # albumentatons: normalized [x_min, y_min, x_max, y_max]
    # coco: [x_min, y_min, width, height]
    # yolo: normalized [x_center, y_center, width, height]
    if source_fmt != 'albumentations':
        bboxes = convert_bboxes_to_albumentations(bboxes, 
            source_fmt, rows=h, cols=w)
    bboxes = convert_bboxes_from_albumentations(bboxes,
        target_fmt, rows=h, cols=w)
    return bboxes 
