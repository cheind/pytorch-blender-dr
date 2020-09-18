import re
import numpy as np
import bpy
from blendtorch import btb

reg = r'^obj\_(\d+)\.'

def slice_from_rect(minc, wh):
    r = (
        slice(minc[1], minc[1]+wh[1],1), 
        slice(minc[0], minc[0]+wh[0],1)
    )
    return r

def bboxes(cam, objs, simplified_geo):
    '''Return bounding boxes in pixel space.'''
    bboxes = []
    for obj in objs:
        xyz = btb.utils.dehom(btb.utils.hom(simplified_geo[obj[1]]) @ np.asarray(obj[0].matrix_world).T)
        xy = cam.ndc_to_pixel(cam.world_to_ndc(xyz))

        #xy = cam.object_to_pixel(obj[0])
        xy = np.concatenate(
            (np.clip(xy[...,0:1],0,cam.shape[1]-1),
            np.clip(xy[...,1:2],0,cam.shape[0]-1)),
            -1
        )
        minc = xy.min(0).astype(np.int32)
        maxc = xy.max(0).astype(np.int32)
        bboxes.append([minc[0], minc[1], maxc[0]-minc[0], maxc[1]-minc[1]])        
    return np.stack(bboxes)


    # bbox_xy = cam.bbox_object_to_pixel(*objs, return_depth=False)
    # bbox_xy = bbox_xy.reshape(-1,8,2)
    # # Clip corners to image size
    # bbox_xy = np.concatenate(
    #     (np.clip(bbox_xy[...,0:1],0,cam.shape[1]-1),
    #     np.clip(bbox_xy[...,1:2],0,cam.shape[0]-1)),
    #     -1
    # )
    # minc = bbox_xy.min(1).astype(np.int32)
    # maxc = bbox_xy.max(1).astype(np.int32)
    # wh = maxc-minc
    # return np.concatenate((minc,wh), -1)
    
def compute_visfracs(cam, objs, bboxes):
    visfrac = np.zeros(len(objs), dtype=np.float32)

    xy, wh = bboxes[:, :2], bboxes[:, 2:]    
    areas = np.prod(wh,1)

    # Find all with non-zero area and center infront of camera
    mask = areas > 0
    ids = np.where(mask)[0]
    for idx in ids:
        visfrac[idx] = btb.utils.compute_object_visibility(objs[idx][0], cam, N=50, dist=20.)
    return visfrac


def classids(objs):
    '''Return class indices of given objects.'''
    return np.array(
        [int(re.match(reg, obj.name).group(1)) for obj,c in objs],
        dtype=np.long)



    




