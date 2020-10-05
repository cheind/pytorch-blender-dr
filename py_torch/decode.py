import torch
import torch.nn.functional as F


def _nms(heat, kernel=3):
    # select padding to keep map dimensions
    pad = (kernel - 1) // 2
    # due to stride 1 and kernel size 3 a peak can be present
    # at most 9 times in hmax and can effect a 5x5 area of heat
    # => in this way we create a more sparse heat map as we would
    # get if a 3x3 kernel with stride 3 was used!
    # this nms method can detect with a minimum distance of 2
    # pixels in between center point detection peaks
    hmax = F.max_pool2d(heat, kernel, 1, pad)
    keep = (hmax == heat).float()  # zero-one mask
    # largest elements remain, others are zero
    return heat * keep  # keeps heat dimensions!


def _topk(heat: torch.Tensor, k):
    """
    Enhances the torch.topk version
    torch.topk(input, k, dim=-1) -> (values, indices)
    indices of the input tensor, values and indices
    are sorted in descending order!

    Parameters
    ----------
    heat: b x c x h x w
        model output heat map
    k: int
        find the k best model output positions

    Returns
    -------
    (topk_score, topk_inds, topk_cids, topk_ys, topk_xs)

    topk_score: b x k
        scores are values form peaks in h x w plane, over multiple
        classes c(or channels)
    topk_inds: b x k
        indices values [0, h * w), over multiple classes
    topk_cids: b x k
        [0, num. of classes), class index to which the score and inds
        belong

    -> each entry in inds can be given as (x, y) coordinate tuple
    topk_ys: b x k
    topk_xs: b x k
    """

    batch, cat, height, width = heat.size()
    topk_scores, topk_inds = torch.topk(heat.view(batch, -1), k)
    topk_cids = torch.true_divide(topk_inds, (height * width)).int()
    topk_inds = topk_inds % (height * width)
    topk_ys = torch.true_divide(topk_inds, width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_cids, topk_ys, topk_xs


def _gather_feat(feat, ind):
    """

    Parameters
    ----------
    feat: b x h * w x c
    ind: b x k

    Returns
    -------
    a x d x c
    """
    dim = feat.size(2)  # c
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # b x k x c
    feat = feat.gather(1, ind)  # b x k x c

    return feat


def _transpose_and_gather_feat(feat, ind):
    """
    If the network output is given as b x c x h x w and the
    indices at which we want to look at [0, h * w) with shape
    of b x k -> return b x k x c which are the entries for each
    channel and batch of the feature map h x w at the given indices!

    Parameters
    ----------
    feat: b x c x h x w
    ind: b x k

    Returns
    -------
    b x k x c, where k indices are provided for h * w values
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()  # b x h x w x c
    feat = feat.view(feat.size(0), -1, feat.size(3))  # b x h * w x c
    feat = _gather_feat(feat, ind)  # b x k x c
    return feat  # b x k x c


def decode(out, k):
    """
    From network output to center point detections.

    Parameters
    ----------
    out: model output
        cpt_hm: b x c x h x w
            where c is the num. of object classes, center point map
        wh: b x 2 x h x w
            width, height prediction map of bounding box
        cpt_off: b x 2 x h x w
            offset prediction map cause by going discrete
    k: scalar
        top k of center point peaks are considered

    Returns
    -------
    tensor b x k x 6 is a concatenation of:
        topk_bboxes b x k x 4,
        topk_scores b x k x 1,
        topk_class number b x k x 1
    ...in exact that order!
    """
    cpt_hm = torch.sigmoid(out["cpt_hm"])
    cpt_off = out["cpt_off"]
    wh = out["wh"]

    b = cpt_hm.size(0)
    cpt_hm = _nms(cpt_hm)  # b x c x h x w

    # each of shape: b x k
    topk_scores, topk_inds, topk_cids, topk_ys, topk_xs = _topk(cpt_hm, k)

    topk_cpt_off = _transpose_and_gather_feat(cpt_off, topk_inds)  # b x k x 2

    # each of shape: b x k
    topk_xs = topk_xs.view(b, k, 1) + topk_cpt_off[..., 0:1]
    topk_ys = topk_ys.view(b, k, 1) + topk_cpt_off[..., 1:2]

    topk_wh = _transpose_and_gather_feat(wh, topk_inds)  # b x k x 2
    topk_cids = topk_cids.view(b, k, 1).float()  # b x k x 1
    topk_scores = topk_scores.view(b, k, 1)  # b x k x 1

    # bboxes, coco format: x, y, width, height; b x k x 4
    topk_bboxes = torch.cat([topk_xs - topk_wh[..., 0:1] / 2,
                             topk_ys - topk_wh[..., 1:2] / 2,
                             topk_wh[..., 0:1],
                             topk_wh[..., 1:2]], dim=-1)
    detections = torch.cat([
        topk_bboxes,
        topk_scores,
        topk_cids,
    ], dim=2)  # b x k x 6

    # for each item in the batch return the top k bboxes together
    # with the corresponding scores and class ids
    return detections  # b x k x 6


def filter_dets(dets, thres):
    """
    Parameters
    ----------
    dets: b x k x 6
    thres: scalar
    """
    b = dets.size(0)
    scores = dets[..., 4]  # b x k
    
    mask = scores >= thres  # b x k
    filtered_dets = dets[mask]  # b * k_filtered x 6
    return filtered_dets.view(b, -1, 6)
