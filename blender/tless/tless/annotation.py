import re
import numpy as np

reg = r'^obj\_(\d+)\.'

def slice_from_rect(minc, wh):
    r = (
        slice(minc[1], minc[1]+wh[1],1), 
        slice(minc[0], minc[0]+wh[0],1)
    )
    return r

def bboxes(cam, objs):
    '''Return bounding boxes in pixel space.'''
    bbox_xy, bbox_z = cam.bbox_object_to_pixel(*objs, return_depth=True)
    bbox_xy = bbox_xy.reshape(-1,8,2)
    bbox_z = bbox_z.reshape(-1,8)

    # Clip corners to image size
    bbox_xy = np.concatenate(
        (np.clip(bbox_xy[...,0:1],0,cam.shape[1]-1),
        np.clip(bbox_xy[...,1:2],0,cam.shape[0]-1)),
        -1
    )
    minc = bbox_xy.min(1).astype(np.int32)
    maxc = bbox_xy.max(1).astype(np.int32)
    wh = maxc-minc
    return np.concatenate((minc,wh), -1), bbox_z
    
   
def compute_visfracs(cam, bbox_obj, bbox_obj_z, bbox_occ, bbox_occ_z):
    visfrac = np.ones(len(bbox_obj), dtype=np.float32)

    minc_obj, wh_obj = bbox_obj[:, :2], bbox_obj[:, 2:]
    minc_occ, wh_occ = bbox_occ[:, :2], bbox_occ[:, 2:]
    areas_obj = np.prod(wh_obj,1)
    z_obj = bbox_obj_z.min(1)
    z_occ = bbox_occ_z.min(1)
    
    # Find all with non-zero area and center infront of camera
    mask = np.logical_and(areas_obj>0, z_obj>0)
    ids = np.where(mask)[0]
    # Create artifical approximate! depth map
    dmap = np.full(cam.shape, 1000., dtype=np.float32)
    for idx in ids:
        r = slice_from_rect(minc_obj[idx], wh_obj[idx])
        dmap[r] = np.minimum(z_obj[idx].reshape(1,1), dmap[r])
    # Account for occluders
    for xy,wh,z in zip(minc_occ,wh_occ,z_occ):
        r = slice_from_rect(xy, wh)
        dmap[r] = np.minimum(z.reshape(1,1), dmap[r])


    # Compute visibility fraction
    for idx in ids:
        r = slice_from_rect(minc_obj[idx], wh_obj[idx])
        visfrac[idx] = (abs(dmap[r] - z_obj[idx])<1e-3).sum() / areas_obj[idx]
    # For all others
    visfrac[~mask] = 0.

    return visfrac


def classids(objs):
    '''Return class indices of given objects.'''
    return np.array(
        [int(re.match(reg, obj.name).group(1)) for obj in objs],
        dtype=np.long)