import json
import yaml
import numpy as np
import sys
import os
from pytorch.utils import data

from .main import MAPPING

class TLessDataset(data.Dataset):

    def __init__(self, opt, item_transform):
        if opt.train:
            self.base_path = Path(opt.train_path)
        else:
            self.base_path = Path(opt.inference_path)
        assert self.basepath.exists()

        self.item_transform = item_transform
        self.opt = opt

        scenes = [f for f in self.basepath.iterdir() if f.is_dir()]
        for scenepath in scenes:
            is_bop_format = (scenepath / 'scene_gt.json').exists()
            is_old_format = (scenepath / 'gt.yml').exists()
            assert is_bop_format or is_old_format, 'Cannot determine format.'
            
            if is_bop_format:
                logging.info('Dataset loading BOP format')
                rgbpaths, bboxes, cids = self._parse_bop_scene(scenepath)
            else:
                logging.info('Dataset loading OLD format')
                rgbpaths, bboxes, cids = self._parse_old_format(scenepath)
                
            self.all_rgbpaths = rgbpaths
            self.all_bboxes = bboxes
            self.all_cids = cids
         
        # a unique id per image path
        self.img_ids = list(range(len(self.all_rgbpaths)))

        # remap class ids to groups
        for i in range(len(self.all_cids)):
            # take the old id and map it to a new one
            new_ids = [MAPPING[old_id] for old_id in self.all_cids[i]]
            self.all_cids[i] = np.array(new_ids, dtype=np.int32)

        if not opt.train:
            # produce ground truth annotations
            rbg_relpaths = [p.relative_to(self.basepath) for p in self.all_rgbpaths]
            create_gt_anns(rbg_relpaths, self.all_bboxes, self.all_cids, 
                "./evaluation/gt.json")
              
    def __len__(self):
        return len(self.all_rgbpaths)
    
    def __getitem__(self, index: int):
        image = cv2.imread(str(self.all_rgbpaths[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        item = {
            "image": image,  # np.ndarray, h x w x 3
            "bboxes": self.all_bboxes[index],
            "cids": self.all_cids[index],
        }
        if not self.opt.train:
            item["image_id"] = index
        item = self.item_transform(item)
        return item

    def _parse_bop_scene(self, scenepath):
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
            assert len(paths) == 1
            rgbpath = paths[0]
            
            cids = [int(e['obj_id']) for e in scene_gt[idx]]
            bboxes = [e['bbox_visib'] for e in scene_gt_info[idx]]
            visib_fracts = [e['visib_fract'] for e in scene_gt_info[idx]]
            
            # filter bboxes by visibility
            filtered_bboxes, filtered_cids = [], []
            for bbox, cid, vis in zip(bboxes, cids, visib_fracts):
                if vis > self.opt.bbox_visibility_threshold:
                    filtered_bboxes.append(bbox)
                    filtered_cids.append(cid)

            # filter out empty images (no bboxes)
            if len(filtered_bboxes) > 0:                  
                all_rgbpaths.append(rgbpath)
                # list of n_objs x 4
                all_bboxes.append(np.array(filtered_bboxes))
                # list of n_objs,
                all_clsids.append(np.array(filtered_cids))
            
        return all_rgbpaths, all_bboxes, all_clsids
    
    def _parse_old_format(self, scenepath):   
        with open(scenepath / 'gt.yml', 'r') as fp:            
            scene_gt = yaml.load(fp.read(), Loader=yaml.Loader)
            
        all_rgbpaths = []
        all_bboxes = []
        all_clsids = []
        
        for idx in scene_gt.keys():
            rgbpath = scenepath / 'rgb' / f'{int(idx):04d}.png'
            assert rgbpath.exists()
            
            cids = [int(e['obj_id']) for e in scene_gt[idx]]
            bboxes = [e['obj_bb'] for e in scene_gt[idx]]
                       
            all_rgbpaths.append(rgbpath)
            all_bboxes.append(np.array(bboxes))
            all_clsids.append(np.array(cids))
            
        return all_rgbpaths, all_bboxes, all_cids
