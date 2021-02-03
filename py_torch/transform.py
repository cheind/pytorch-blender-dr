import albumentations as A
import numpy as np 

from .main import MAPPING

def item_filter(func, bbox_visibility_threshold):

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

class TrainTransformation:

    def __init__(self, opt):
        self.h, self.w = opt.h, opt.w  # e.g. 512 x 512
        self.num_classes = opt.num_classes  # num. of object classes
        self.n_max = opt.n_max  # num. of max objects per image
        self.down_ratio = opt.down_ratio  # => low resolution 128 x 128
        self.mean = opt.mean
        self.std = opt.std

        transformations = [
            A.SmallestMaxSize(max_size=min(opt.h, opt.w)),
            A.RandomCrop(height=opt.h, width=opt.w) if opt.augment else A.CenterCrop(height=opt.h, width=opt.w),
        ]

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

