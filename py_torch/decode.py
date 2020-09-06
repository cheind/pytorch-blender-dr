import torch
import numpy as np


def decode(out, thres=0.3):  # TODO
    """
    model output:
    cpt_hm batch x num_classes x h x w
    cpt_off: batch x 2 x h x w
    wh: batch x 2 x h x w

    goal, decoded to:
    bboxes [[x, y, w, h],...]
    where the center point scores are >= thres
    """
    cpt_hm = out["cpt_hm"]
    cpt_off = out["cpt_off"]
    wh = out["wh"]


    return bboxes  # batch x 4