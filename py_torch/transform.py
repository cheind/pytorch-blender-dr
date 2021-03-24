import albumentations as A
import numpy as np 
from typing import List
import cv2
import os 
from glob import glob
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

from .utils import generate_heatmap
from .constants import MAPPING

def load_fda_images(fda_path, rank=0, cache=False, max_gb=2.0):
    # returns None or list of path strings or numpy images if cache=True!
    # Caching images (11.757GB): 100%|██████████████████| 10080/10080 [00:44<00:00, 226.40it/s]

    if not os.path.exists(fda_path) and rank == 0:
        print('Invalid folder path for Fourier Domain Adaptation(FDA) images:', fda_path)
    else:
        # load tless RGB images
        extensions = (
            "*.png", 
            "*.jpg", 
            "*.jpeg",
        )
        fda_files = [] 
        for ext in extensions:
            # there are also subfolders with mask, depth we don't want!
            pattern = os.path.join(fda_path, '*' ,'rgb', ext)
            fda_files.extend(glob(pattern, recursive=True))

        # import pdb; pdb.set_trace()

        if len(fda_files) > 0:
            if cache:
                fda_files = np.random.permutation(fda_files)  # shuffle randomly
                results = ThreadPool(4).imap(read_fn, fda_files)
                if rank == 0:
                    pbar = tqdm(results, total=len(fda_files))
                else:
                    pbar = results

                gb = 0
                fda_images = []
                for image in pbar:
                    fda_images.append(image)
                    gb += image.nbytes
                    pbar.desc = f'Caching images ({gb / 1E9:.3f}GB)'
                    if gb/1E9 >= max_gb:
                        if rank == 0:
                            print(f'Reached maximum of allowed caching memory: {max_gb}GB')
                        break
                    
                return fda_images  # numpy images
            else:
                return fda_files  # list of path strings

    return None  # otherwise

def read_fn(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def identity_fn(x):
    return x

class Transform:

    def __init__(self, opt, bbox_format='coco',
        augmentations: List = None, normalize=True,
        resize_crop=False, bboxes=True, vis_filter=False):
        super().__init__()
        self.h, self.w = opt.h, opt.w  # e.g. 512 x 512
        self.classes = opt.classes  # num. of object classes
        self.n_max = opt.n_max  # num. of max objects per image
        self.down_ratio = opt.down_ratio  # => low resolution 128 x 128
        self.sigma = opt.sigma
        self.normalize_wh = opt.normalize_wh
        self.mean = opt.mean
        self.std = opt.std
        self.train = opt.train 
        self.resize_crop = resize_crop
        self.normalize = normalize
        self.augmentations = augmentations
        self.bbox_format = bbox_format 
        self.bboxes = bboxes
        self.vis_filter = vis_filter
        self.bbox_visibility_threshold = opt.bbox_visibility_threshold

        self.cache = opt.cache_fda_images
        self.fda_only = opt.fda_only
        self.fda_images = load_fda_images(opt.fda_path, opt.rank, cache=self.cache)

        if opt.train:
            if augmentations is None:
                # default augmentation if not given explicitly
                augmentations = [
                    A.HueSaturationValue(p=0.5),
                    A.ChannelShuffle(p=0.4),
                    A.HorizontalFlip(p=0.2),
                    A.GaussNoise(p=0.3),
                ]

            if self.fda_images is not None:
                if self.fda_only:
                    transformations = [
                        A.FDA(self.fda_images, p=0.5, read_fn=identity_fn if self.cache else read_fn)
                    ]
                else:  # combine FDA with std augmentation pipeline
                    transformations = [
                        A.OneOrOther(
                            A.Sequential(augmentations),
                            A.FDA(self.fda_images, 
                                read_fn=identity_fn if self.cache else read_fn),
                        p=1)
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
            ]
            bbox_params = None

        if normalize:  # zero mean, unit variance
            transformations.append(A.Normalize(mean=opt.mean, std=opt.std))

        self.transform_fn = A.Compose(transformations, 
            bbox_params=bbox_params)

        if opt.world_size == 1:  # non mp case!
            if opt.train:
                self.item_transform = self.item_transform_train
            else:
                self.item_transform = self.item_transform_test

            if vis_filter:
                self.item_transform = self.item_filter(self.item_transform,
                    opt.bbox_visibility_threshold)   
            
    def item_transform_mp(self, item: dict):
        '''
        A transformation without nested structure to be serializeable
        in a multiprocessing setting!
        '''
        if self.vis_filter:
            bboxes = item['bboxes']  # n x 4
            cids = item['cids']  # n,
            vis = item['visfracs']  # n,

            # visibility filtering
            mask = (vis > self.bbox_visibility_threshold)  # n,
            item["bboxes"] = bboxes[mask]  # n' x 4
            cids = cids[mask]  # n',

            # remap the class ids to our group assignment
            new_ids = np.array([MAPPING[old_id] for old_id in cids], dtype=cids.dtype)
            item["cids"] = new_ids
        
        if self.train:  
            item = self.item_transform_train(item)
        else:
            item = self.item_transform_test(item)
        return item
            
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
            - bboxes: n x 4; [[x, y, width, height], ...]
            - cpt_hm: 1 x 128 x 128 # classes x 128 x 128
            - cpt_off: n_max x 2 low resolution offset - [0, 1)
            - cpt_ind: n_max, low resolution indices - [0, 128^2)
            - cpt_mask: n_max,
            - wh: n_max x 2, low resolution width, height - [0, 128], [0, 128]
            - cids: n_max,
        """
        image = item["image"]  # h x w x 3
        bboxes = item['bboxes']  # n x 4
        cids_old = item["cids"]  # n,
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

        # the original size of bboxes is in [0, h], [0, w] range
        if not self.normalize_wh:  # prepare targets for bbox regression
            # LOW RESOLUTION in range [0, hl] and [0, wl]
            # for height and width respectively
            wh = np.zeros((self.n_max, 2), dtype=np.float32)
            wh[:len_valid, :] = bboxes[:, 2:-1] / self.down_ratio
        else:
            # NORMALIZED in range [0, 1] and [0, 1]
            # for height and width respectively
            wh = np.zeros((self.n_max, 2), dtype=np.float32)
            # COCO bboxes xmin, ymin, width, height
            wh[:len_valid, 0] = bboxes[:, 2] / self.w
            wh[:len_valid, 1] = bboxes[:, 3] / self.h
            
        cids = np.zeros((self.n_max,), dtype=np.uint8)
        # the bbox labels help to reassign the correct classes
        cids[:len_valid] = cids_old[bboxes[:, -1].astype(np.int32)]

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
        for i in range(self.classes):
            mask = (cids[:len_valid] == i)  # n_valid,
            xy = valid_cpt[mask]  # n x 2, valid entries for each class
            # each generated heatmap has 1 x hl x wl
            cpt_hms.append(generate_heatmap((hl, wl), xy, sigma=self.sigma))  
        
        cpt_hm = np.concatenate(cpt_hms, axis=0) 

        # same shape for default collate_fn
        bboxes_ = np.zeros((self.n_max, 4))
        # high resolution bboxes
        bboxes_[:len_valid, :] = bboxes[:, :4]
        
        item = {
            "image": image,
            "bboxes": bboxes_,
            "cpt_hm": cpt_hm,
            "cpt_off": cpt_off,
            "cpt_ind": cpt_ind,
            "cpt_mask": cpt_mask,
            "wh": wh,
            "cids": cids,
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
