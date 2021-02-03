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

from .utils import Config
from .model import get_model
from .decode import decode, filter_dets
from .visu import render
from .main import MAPPING, item_filter
from .evaluation import create_gt_anns, evaluate


def main(opt):
    transformation = Transformation(opt)
    item_transform = transformation.item_transform

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

    
    
if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)

    opt = Config("./configs/config.txt")
    print(opt)

    main(opt)


