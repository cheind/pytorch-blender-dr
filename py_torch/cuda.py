from __future__ import print_function
from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import sys
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')
from subprocess import call

print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('Active CUDA Device: GPU', torch.cuda.current_device())

print('Available devices ', torch.cuda.device_count())
print('Current cuda device ', torch.cuda.current_device())

model = torch.nn.Linear(4, 5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
