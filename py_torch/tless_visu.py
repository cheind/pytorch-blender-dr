from torch.utils import data
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import cv2


from .visu import tless_visu 
from .main import CLSES_MAP


class TLessDataset(data.Dataset):

    def __init__(self, basepath):

        self.basepath = Path(basepath)
        assert self.basepath.exists()
        
        # 000001, 000002,...
        self.all_rgbpaths = []
        self.all_bboxes = []
        self.all_clsids = []
        self.all_bboxes_visib = []
        self.all_visib_fracts = []
        
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

                bboxes_visib = [e['bbox_visib'] for e in scene_gt_info[idx]]
                visib_fracts = [e['visib_fract'] for e in scene_gt_info[idx]]
                
                self.all_rgbpaths.append(rgbpath)
                # list of n_objs x 4
                self.all_bboxes.append(np.array(bboxes))
                self.all_bboxes_visib.append(np.array(bboxes_visib))
                self.all_visib_fracts.append(np.array(visib_fracts))

                self.all_clsids.append(np.array(clsids))

        # create image ids for evaluation, each image path has 
        # a unique id
        self.img_ids = list(range(len(self.all_rgbpaths)))

        # remap class ids to groups
        for i in range(len(self.all_clsids)):
            # take the old id and map it to a new one
            new_ids = [CLSES_MAP[old_id] for old_id in self.all_clsids[i]]
            self.all_clsids[i] = np.array(new_ids, dtype=np.int32)

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
            "bboxes_visib": self.all_bboxes_visib[index],
            "visib_fracts": self.all_visib_fracts[index]
        }
        return item


if __name__ == "__main__":
    path = "/mnt/data/tless_train_pbr"
    path = "/mnt/data/tless_test_real/test_primesense"
    ds = TLessDataset(path)

    for step, item in enumerate(ds):
        tless_visu(item, step)
        
        if step > 100:
            break