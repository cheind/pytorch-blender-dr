import re
import numpy as np

reg = r'^obj\_(\d+)\.'

def bboxes(cam, objs):
    '''Return bounding boxes in pixel space.'''
    bbox_xy = cam.bbox_object_to_pixel(*objs).reshape(-1,8,2)
    minc = bbox_xy.min(1)
    maxc = bbox_xy.max(1)
    wh = maxc-minc
    return np.concatenate((minc,wh), -1)

def classids(objs):
    '''Return class indices of given objects.'''
    return np.array(
        [int(re.match(reg, obj.name).group(1)) for obj in objs],
        dtype=np.long)