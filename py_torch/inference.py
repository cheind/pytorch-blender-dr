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




def main(opt):
    

    
    
if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)

    opt = Config("./configs/config.txt")
    print(opt)

    main(opt)


