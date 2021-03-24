from collections import defaultdict
import sys
import io
import os
from tqdm import tqdm
import math
import torch
from torch import nn
import numpy as np
import thop
import time
from copy import deepcopy
from torchvision import ops
import contextlib
from typing import Dict, List, Tuple
import torch.distributed as dist
from torch.cuda.amp import autocast
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune

def autopad(k, p=None):
    if p is None:  # pad s.t. same spatial shape after convolution
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers 
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    if isinstance(conv, nn.Conv2d):
        conv_type = nn.Conv2d
    elif isinstance(conv, ops.DeformConv2d):
        conv_type = ops.DeformConv2d
    
    fusedconv = conv_type(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)
        
    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

@torch.no_grad()
def profile(model, input_size=512, nruns=100, verbose=False, amp=False):
    # used to benchmark inference speed of model on different devices

    if not (isinstance(input_size, tuple) or isinstance(input_size, list)):
        input_size = (input_size, input_size)
    x = torch.randn(1, 3, *input_size)
    
    model.eval()
    param = next(model.parameters())
    device = param.device
    x = x.to(device)

    if param.is_cuda:
        if torch.backends.cudnn.benchmark:
            # have to do warm up iterations for fair comparison
            print('benchmark warm up...')
            with autocast(enabled=amp):
                for _ in range(50):
                    _ = model(x)
        
        start = time_synchronized()
        with autocast(enabled=amp):
            for _ in range(nruns):
                o = model(x)
        end = time_synchronized()
        print(f'Forward time: {(end - start) * 1000 / nruns:.3f}ms (cuda)',
            '@ input:', x.shape)
    else:
        start = time_synchronized()
        for _ in range(nruns):
            o = model(x)
        end = time_synchronized()  # seconds
        print(f'Forward time: {(end - start) * 1000 / nruns:.3f}ms (cpu)',
            '@ input:', x.shape)
    
    if verbose:
        if isinstance(o, dict):
            for head_key, head in o.items():
                print(f'{head_key} output: {head.size()}')
        elif isinstance(o, list) or isinstance(o, tuple):
            print('output:', end=' ')
            for head in o:
                print(head.size(), end=', ')
            print('')
        else:
            print('output:', o.size())
            
def profile_training(model, input_size=512, nruns=100, amp=False, batch_size=16):
    if not (isinstance(input_size, tuple) or isinstance(input_size, list)):
        input_size = (input_size, input_size)
    x = torch.randn(batch_size, 3, *input_size)

    assert torch.cuda.is_available()
    model.cuda().train()
    x = x.cuda()
    
    o = model(x)
    if isinstance(o, list) or isinstance(o, tuple):
        g0 = [torch.rand_like(item) for item in o]
    elif isinstance(o, dict):
        g0 = [torch.rand_like(item) for item in o.values()]
    else:
        g0 = [torch.rand_like(o)]
    
    if torch.backends.cudnn.benchmark:
        # have to do warm up iterations for fair comparison
        print('benchmark warm up forward...')
        with torch.no_grad():
            with autocast(enabled=amp):
                for _ in range(50):
                    o = model(x)

        print('benchmark warm up backward...')
        with autocast(enabled=amp):
            for _ in range(50):
                o = model(x)
                for param in model.parameters():
                    param.grad = None
                o = o.values() if isinstance(o, dict) else ([o] if isinstance(o, torch.Tensor) else o)         
                for i, v in enumerate(o):
                    v.backward(g0[i], retain_graph=i < len(o) - 1)

    print(f'run through forward pass for {nruns} runs...')
    start = time_synchronized()
    with torch.no_grad():
        with autocast(enabled=amp):
            for _ in range(nruns):
                o = model(x)
    end = time_synchronized()
    fwd_time = end - start  # fwd only
    
    print(f'run through forward and backward pass for {nruns} runs...')
    torch.cuda.reset_peak_memory_stats(device='cuda')
    start = time_synchronized()
    with autocast(enabled=amp):
        for _ in range(nruns):
            o = model(x)
            for param in model.parameters():
                param.grad = None
            o = o.values() if isinstance(o, dict) else ([o] if isinstance(o, torch.Tensor) else o)          
            for i, v in enumerate(o):
                v.backward(g0[i], retain_graph=i < len(o) - 1)
    end = time_synchronized()
    mem = torch.cuda.max_memory_reserved(device='cuda')  # bytes
    bwd_time = end - start  # fwd + bwd
    bwd_time = (bwd_time - fwd_time)  # bwd only

    print(f'Forward time: {fwd_time * 1000 / nruns:.3f}ms (cuda)',
        '@ input:', x.shape)
    print(f'Backward time: {bwd_time * 1000 / nruns:.3f}ms (cuda)', 
        '@ input:', x.shape)
    print(f'Maximum of managed memory: {mem / 10**9}GB')

def init_torch_seeds(seed=0, verbose=False):
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False
    
    if verbose:
        print('PyTorch version {}'.format(torch.__version__))
        print('CUDA version {}'.format(torch.version.cuda))
        print('cuDNN version {}'.format(cudnn.version()))
        print('cuDNN deterministic {}'.format(cudnn.deterministic))
        print('cuDNN benchmark {}'.format(cudnn.benchmark))

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # not applicable for Center Net since parts have special initialization
        elif t is nn.BatchNorm2d:
            pass
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True
            
def model_info(model, in_channels=3, in_height=512, in_width=512, verbose=False):
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients

    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    x = torch.randn(1, in_channels, in_height, in_width, 
        device=next(model.parameters()).device)
    # macs ... multiply-add computations
    # flops ... floating point operations
    macs, _ = thop.profile(deepcopy(model), inputs=(x,), verbose=False)
    flops = macs / 1E9 * 2  # each mac = 2 flops (addition + multiplication)

    n_layers = len(list(model.modules()))
    print(f'{model.__class__.__name__}: {n_layers} layers, {n_p/10**6:0.3}M parameters, {n_g/10**6:0.3}M gradients, {flops:.1f}GFLOPs')
    return n_layers, n_p, n_g, flops

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()  # seconds

def setup(rank, world_size):
    # A free port on the machine that will host the process with rank 0.
    os.environ['MASTER_ADDR'] = 'localhost'
    # IP address of the machine that will host the process with rank 0.
    os.environ['MASTER_PORT'] = '12355'
    # The total number of processes, so master knows how many workers to wait for.
    os.environ['WORLD_SIZE'] = str(world_size)
    # Rank of each process, so they will know whether it is the master of a worker.
    os.environ['RANK'] = str(rank)
    # initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def _to_float(x):
    return float(f"{x:.2f}")

def sparsity(model):
    # Return global model sparsity
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a

def prune_weights(model, amount=0.1):
    # Prune model to requested global sparsity
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print('==> %.3g global sparsity' % sparsity(model).item())

def item_transform_image_only(item: Dict):
    # to yield images for dataset statistic calculation
    image = item['image']  # [0, 255] and unit8
    image = image / 255.0  # [0, 1] and float32
    return {'image': image}

def data_mean_and_std(dataloader, channel_first=False):
    # calculate statistics of datasets
    # if channel_first: b x c x h x w else: b x h x w x c
    # dataloader should yield non-normalized images with 
    # floating point values in [0, 1] range
    # note: Var[x] = E[X^2] - E^2[X]
    N = 0
    C = next(iter(dataloader))['image'].size(1 if channel_first else 3)
    channelwise_sum = torch.zeros(C)
    channelwise_sum_squared = torch.zeros(C)
    for batch in tqdm(dataloader, desc='gather data statistics'):
        images = batch['image']
        #import pdb; pdb.set_trace()
        if not channel_first:  # from: b x h x w x c
            images = images.permute(0, 3, 1, 2)  # to: b x c x h x w

        N += images.size(0) * images.size(2) * images.size(3) # pixels per channel
        channelwise_sum += images.sum([0, 2, 3])  # C,
        channelwise_sum_squared += torch.square(images).sum([0, 2, 3])  # C,
    
    mean = channelwise_sum / N  # C,
    std = torch.sqrt(channelwise_sum_squared / N - torch.square(mean))  # C,
    return mean, std

def generate_heatmap(shape, xy: np.ndarray, mask=None, sigma=2, cutoff=1e-3, bleed=True):
    """
    Generates a single belief map of 'shape' for each point in 'xy'.

    Parameters
    ----------
    shape: tuple
        h x w of image
    xy: n x 2
        n points with x, y coordinates (image coordinate system)
    mask: n,
        zero-one mask to select points from xy
    sigma: scalar
        gaussian sigma
    cutoff: scalar
        set belief to zero if it is less then cutoff

    Returns
    -------
    belief map: 1 x h x w
    """
    n = xy.shape[0]
    h, w = shape[:2] 

    if n == 0:
        return np.zeros((1, h, w), dtype=np.float32)

    if not bleed:
        wh = np.asarray([w - 1, h - 1])[None, :]
        mask_ = np.logical_or(xy[..., :2] < 0, xy[..., :2] > wh).any(-1)
        xy = xy.copy()
        xy[mask_] = np.nan

    # grid is 2 x h x h
    grid = np.array(np.meshgrid(np.arange(w), np.arange(h)), dtype=np.float32)
    # reshape grid to 1 x 2 x h x w
    grid = grid.reshape((1, 2, h, w))
    # reshape xy to n x 2 x 1 x 1
    xy = xy.reshape((n, 2, 1, 1))
    # compute squared distances to joints
    d = ((grid - xy) ** 2).sum(1)
    # compute gaussian
    b = np.nan_to_num(np.exp(-(d / (2.0 * sigma ** 2))))
    b[(b < cutoff)] = 0  # b is n x h x w

    if mask is not None:
        # set the invalid center point maps to all zero
        b *= mask[:, None, None]  # n x h x w

    b = b.max(0, keepdims=True)  # 1 x h x w
    b[b >= 0.95] = 1  # targets are exactly 1 at discrete positions 
    return b  # 1 x h x w

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # from https://github.com/ultralytics/yolov5/blob/master/utils/general.py
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T  # 4xn

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    #import pdb; pdb.set_trace()
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                # nan for perfect alignment!
                #return torch.nan_to_num(iou - (rho2 / c2 + v * alpha), nan=1.0)  # CIoU
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MetricMeter:

    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                'Input to MetricMeter.update() must be a dictionary'
            )

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def reset(self):  # to clear history for average calculation
        for meter in self.meters.values():  # each of type AvarageMeter
            meter.reset()

    def to_writer(self, writer, tag, n_iter):
        for name, meter in self.meters.items():
            writer.add_scalar(f"{tag}/{name}", meter.val, n_iter)

    def get_avg(self, tag):
        return self.meters[tag].avg

    def get_val(self, tag):
        return self.meters[tag].val

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(
                '{} {:.4f} ({:.4f})'.format(name, meter.val, meter.avg)
            )
        return self.delimiter.join(output_str)

class Config:

    def __init__(self, config_path):
        with open(config_path, "r") as fh:
            content = fh.read()
        self.parse(content)

    def parse(self, content):
        for line in content.split('\n'):
            if len(line) == 0 or line.startswith('#'):
                continue  # skip comments and empty lines
            try:
                k, v = line.split(':')
            except ValueError as e:
                print(e, 'error in line:', line)
                raise AttributeError

            if '[' in v:  # parse lists
                is_float = True if '.' in v else False
                v = v.strip()[1:-1].split(',')
                v = [x for x in v if x != '']  # case: [0, ] 
                v = list(map(float if is_float else int, v))
                dtype = np.float32 if is_float else np.int32
                v = np.array(v, dtype=dtype)
            elif '/' in v or "'" in v or '"' in v:  # parse paths or strings
                v = v.strip().strip("'").strip('"')
            else:  # parse integer, floating point or string values
                is_float = True if '.' in v else False
                try:
                    v = float(v) if is_float else int(v)
                except ValueError:
                    if "True" in v:
                        v = True
                    elif "False" in v:
                        v = False

            setattr(self, k, v)
            
        # import pdb; pdb.set_trace()

    def __repr__(self):
        info = []
        for k, v in self.__dict__.items():
            info.append(f"{k}: {v}")
        return "\n".join(info)

    def __str__(self):
        return self.__repr__()
    
class FileStream:
    # context manager to save print output

    def __init__(self, filepath: str, parser=None):
        self.filepath = filepath
        self.file = None
        self.buffer = io.StringIO()
        self.parser = parser

    def write(self, s):
        self.buffer.write(s)  # redirect to buffer
        sys.__stdout__.write(s)  # and print it to console

    def __enter__(self):
        self.file = open(self.filepath, "w+")
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.parser is not None:
            output = self.parser(self.buffer.getvalue())
        else:
            output = self.buffer.getvalue()
        self.file.write(output)
        self.buffer.close()
        self.file.close()
        sys.stdout = sys.__stdout__
