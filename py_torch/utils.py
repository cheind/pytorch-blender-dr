from collections import defaultdict
import torch
import numpy as np
import sys
import io

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
            k, v = line.split(': ')
            if '[' in v:  # parse lists
                is_float = True if '.' in v else False
                v = v.strip()[1:-1].split(',')
                v = list(map(float if is_float else int, v))
                dtype = np.float32 if is_float else np.int32
                v = np.array(v, dtype=dtype)
            elif '/' in v:  # parse paths
                pass
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

    def __repr__(self):
        info = []
        for k, v in self.__dict__.items():
            info.append(f"{k}: {v}")
        return "\n".join(info)

    def __str__(self):
        return self.__repr__()
    