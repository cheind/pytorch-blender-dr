import torch
import numpy as np

from .loss import _sigmoid


@torch.no_grad()
def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


@torch.no_grad()
def _topk(scores, opt):
    """
    Slow for a large number of classes.

    - scores: batch x num_classes x h x w
    """    
    k = opt.k
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
    topk_clses = (topk_inds / (height * width)).int()
    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


@torch.no_grad()
def decode(out, opt):  # TODO
    """
    model output:
    cpt_hm batch x num_classes x h x w
    cpt_off: batch x 2 x h x w
    wh: batch x 2 x h x w

    goal, decoded to:
    bboxes [[x, y, w, h],...]
    where the center point scores are >= thres
    """
    cpt_hm = _sigmoid(out["cpt_hm"])
    cpt_off = out["cpt_off"]
    wh = out["wh"]

    thres = opt.thres

    topk_scores, topk_inds, topk_clses, topk_ys, topk_xs = topk(scores, opt)

    return bboxes  # batch x 4