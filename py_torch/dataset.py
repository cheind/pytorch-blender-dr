

class TLessTrainDataset(data.Dataset):

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
                logging.info('Dataset loading BOP format')
                rgbpaths, bboxes, clsids = self._parse_bop_scene(scenepath, 
                    opt.bbox_visibility_threshold)
            else:
                logging.info('Dataset loading OLD format')
                rgbpaths, bboxes, clsids = self._parse_old_format(scenepath, 
                    opt.bbox_visibility_threshold)
                
            self.all_rgbpaths.extend(rgbpaths)
            self.all_bboxes.extend(bboxes)
            self.all_cids.extend(clsids)
         
        # a unique id per image path
        self.img_ids = list(range(len(self.all_rgbpaths)))

        # remap class ids to groups
        for i in range(len(self.all_cids)):
            # take the old id and map it to a new one
            new_ids = [MAPPING[old_id] for old_id in self.all_cids[i]]
            self.all_cids[i] = np.array(new_ids, dtype=np.int32)
                
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
        item = self.item_transform(item)
        return item

    def _parse_bop_scene(self, scenepath, bbox_visibility_threshold):
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

            visib_fracts = [e['visib_fract'] for e in scene_gt_info[idx]]
            
            # filter bboxes by visibility
            filtered_bboxes, filtered_cids = [], []
            for bbox, cid, vis in zip(bboxes, cids, visib_fracts):
                if vis > bbox_visibility_threshold:
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
    
    def _parse_old_format(self, scenepath, bbox_visibility_threshold):
        import yaml
        
        with open(scenepath / 'gt.yml', 'r') as fp:            
            scene_gt = yaml.load(fp.read(), Loader=yaml.Loader)
            
        all_rgbpaths = []
        all_bboxes = []
        all_clsids = []
        
        for idx in scene_gt.keys():
            rgbpath = scenepath / 'rgb' / f'{int(idx):04d}.png'
            assert rgbpath.exists()
            
            clsids = [int(e['obj_id']) for e in scene_gt[idx]]
            bboxes = [e['obj_bb'] for e in scene_gt[idx]]
                       
            all_rgbpaths.append(rgbpath)
            all_bboxes.append(np.array(bboxes))
            all_clsids.append(np.array(clsids))
            
        return all_rgbpaths, all_bboxes, all_clsids

class TLessTestDataset(data.Dataset):
    """
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
                visib_fracts = [e['visib_fract'] for e in scene_gt_info[idx]]
                filtered_bboxes, filtered_cids = [], []
               
                # filter bboxes by visibility
                for bbox, cid, vis in zip(bboxes, cids, visib_fracts):
                    if vis > opt.vis_thres:
                        filtered_bboxes.append(bbox)
                        filtered_cids.append(cid)

                # filter out empty images (no bboxes)
                if len(filtered_bboxes) > 0:
                    self.all_rgbpaths.append(rgbpath)
                    # list of n_objs x 4
                    self.all_bboxes.append(np.array(filtered_bboxes))
                    # list of n_objs,
                    self.all_cids.append(np.array(filtered_cids))
        
        # a unique id per image path
        self.img_ids = list(range(len(self.all_rgbpaths)))

        # remap classes
        for i in range(len(self.all_cids)):
            new_ids = [MAPPING[old_id] for old_id in self.all_cids[i]]
            self.all_cids[i] = np.array(new_ids, dtype=np.int32)

        # produce ground truth annotations
        rbg_relpaths = [p.relative_to(self.basepath) for p in self.all_rgbpaths]
        create_gt_anns(rbg_relpaths, self.all_bboxes, self.all_cids, "./evaluation/gt.json")
                
    def __len__(self):
        return len(self.all_rgbpaths)
    
    def __getitem__(self, index: int):
        image = cv2.imread(str(self.all_rgbpaths[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        item = {
            "image": image,  # np.ndarray, h x w x 3
            "bboxes": self.all_bboxes[index],
            "cids": self.all_cids[index],
            "image_id": index,
            }
        item = self.item_transform(item)
        return item

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