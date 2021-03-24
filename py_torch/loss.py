import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import bbox_iou

def _sigmoid(x):
    """
    Clamping values, therefore this activation function can
    be used with Focal Loss.
    """
    y = torch.clamp(torch.sigmoid(x), min=1e-4, max=1-1e-4)
    return y

def gather_from_maps(maps: torch.Tensor, ind: torch.Tensor,
                     mask: torch.Tensor = None):
    """
    Implementation of 'looking through' maps at specified points.
    The points are given as indices from 0 to h * w.
    Where h and w are height and width of the maps.

    Parameters
    ----------
    maps: b x s x h x w
    ind: b x n
    mask: b x n

    Returns
    -------
    b x n x s or
    b x valid n x s (using a mask)

    n ... max num. of objects
    h, w ... low resolution height and width

    Possible calls
    --------------
    center point offset
        map: b x 2 x h x w, ind: b x n

    center point bounding box width and height
        map: b x 2 x h x w, ind: b x n
    """
    maps = maps.permute(0, 2, 3, 1).contiguous()  # b x h x w x s
    maps = maps.view(maps.size(0), -1, maps.size(3))  # b x hw x s
    s = maps.size(2)
    ind = ind.unsqueeze(2).expand(-1, -1, s)  # b x n x s
    maps = maps.gather(1, ind)  # from: b x hw x s to: b x n x s
    if mask is not None:  # b x n
        mask = mask.unsqueeze(2).expand_as(maps)  # b x n x s
        maps = maps[mask]  # b x valid n x s
    return maps  # b x valid n x s or b x n x s

class FocalLoss(nn.Module):

    def __init__(self, alpha=2, beta=4):
        """
        Keep in mind that the given out map, which is the
        network output must be clamped: 0 < p < 1 for each
        pixel values p in the out map!! Otherwise loss is Nan.
        E.g. use clamped sigmoid activation function!

        No mask needed for this loss, the ground truth heat maps
        are produced s.t. all map peaks are valid.

        Parameters
        ----------
        alpha: focal loss parameter from Center Net Paper
        beta: focal loss parameter from Center Net Paper
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, out, target):
        """
        Modified focal loss. Runs faster and costs a little bit more memory.
        The out parameter is the network output which contains
        the center point heat map predictions.

        Parameters
        ----------
        out : (b x num_classes x h x w)
          network output, prediction
        target : (b x num_classes x h x w)
          ground truth heat map

        Returns
        -------
        focal loss : scalar tensor
        """
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        #import pdb; pdb.set_trace()
        pos_weights = torch.pow(1 - out, self.alpha)
        #print(target.shape, out.shape)
        neg_weights = (torch.pow(1 - target, self.beta)
                       * torch.pow(out, self.alpha))

        # masks are zero one masks
        pos_loss = pos_weights * torch.log(out) * pos_mask
        neg_loss = neg_weights * torch.log(1 - out) * neg_mask

        # num. of peaks in the ground truth heat map
        num_p = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        loss = 0
        if num_p == 0:  # no peaks in ground truth heat map
            loss = loss - neg_loss
        else:  # normalize focal loss
            loss = loss - (pos_loss + neg_loss) / num_p
        return loss

class L1Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, out, target, ind, mask):
        """
        Calculate the offset loss (caused by discrete pixels)
        or the size loss (bounding box regression to width and height).

        Parameters
        ----------
        out : b x 2 x h x w
        target : b x n x 2
            ground truth offset of center or key points or
            the bounding box dimensions
        ind : b x n
            indices at which we extract information from out,
            therefore low resolution indices, each index is a scalar
            in range 0 to h * w (index in heat map space)
        mask : b x n
            mask out not annotated or visible key or center points

        n ... max num. of objects
        h, w ... low resolution height and width of feature map

        Returns
        -------
        L1 loss : scalar tensor
        """
        pred = gather_from_maps(out, ind)  # b x n x 2
        mask = mask.unsqueeze(-1)  # b x n x 1
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
        loss = loss / (mask.sum() + 1e-4)
        return loss

class IoULoss(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.GIoU = self.DIoU = self.CIoU = False
        if opt.wh_loss == 'IoU':
            pass
        elif opt.wh_loss == 'GIoU':
            self.GIoU = True
        elif opt.wh_loss == 'DIoU':
            self.DIoU = True
        elif opt.wh_loss == 'CIoU':
            self.CIoU = True

    def forward(self, out, target, ind, mask):
        """
        Calculate the bounding box regression loss using 
        inter section over union functions instead of L1 or L2.

        Parameters
        ----------
        out : b x 2 x h x w
            contains width, height information of bboxes
        target : b x n x 2
            bboxes ground truth width, height
        ind : b x n
            indices at which we extract information from out,
            therefore low resolution indices, each index is a scalar
            in range 0 to h * w (index in heat map space),
        mask : b x n
            mask out not annotated or visible key or center points

        n ... max num. of objects
        h, w ... low resolution height and width of feature map

        Returns
        -------
        IoU loss : scalar tensor
        """
        h, w = out.shape[-2:]  # low res. height, width
        # compute center positions then the upper left bbox corners
        xy_min = torch.stack((ind%w, ind/w), dim=-1).int()  # b x n x 2

        # get low resolution xmin, ymin of bboxes
        # x_min = x_center-w/2, y_min = y_center-h/2,  (w,h of bbox!)
        # since we have a dependecy w.r.t. the used bboxes we have to 
        # distinguish between ground truth(gt) and detection(dt)
        xy_min_gt =  (xy_min - target/2).int()
        # look at gt center positions to obtain wh featues for bboxes!
        detection = gather_from_maps(out, ind)  # b x n x 2
        xy_min_dt =  (xy_min - detection/2).int()
        bboxes_gt = torch.cat((xy_min_gt, target), dim=-1)  # b x n x 4
        bboxes_dt = torch.cat((xy_min_dt, detection), dim=-1)  # b x n x 4

        # due to the usage of the gt center points and selecting the 
        # features for the width and height on this exact point 
        # we have ONE gt bbox facing ONE dt bbox!
        # as a first step we mask out invalid bboxes
        mask = mask.unsqueeze(-1).expand(-1, -1, 4).bool()  # b x n x 4
        # note: for L1 loss it was okay to mask out by setting to 0
        # but here we want to reduce the number of loop iterations done
        # (although it would work if both boxes are [0, 0, 0, 0])
        bboxes_gt = bboxes_gt[mask].reshape(-1, 4)  # mask.sum() x 4
        bboxes_dt = bboxes_dt[mask].reshape(-1, 1, 4)  # mask.sum() x 1 x 4

        loss = 0
        for box1, box2 in zip(bboxes_gt, bboxes_dt):
            # box1 ... 4,
            # box2 ... n x 4  (here n=1)
            iou = bbox_iou(box1, box2, x1y1x2y2=False, GIoU=self.GIoU, DIoU=self.DIoU, CIoU=self.CIoU)
            iou_loss = (1.0 - iou)  # n,
            loss += iou_loss.mean()
        
        loss /= (mask.sum() + 1e-4)
        #import pdb; pdb.set_trace()
        return loss

class CenterLoss(nn.Module):

    def __init__(self, opt):
        super().__init__()
        # set criterion to calculate...
        self.opt = opt
        self.crit_hm = FocalLoss()  # ... loss on heat map
        self.crit_off = L1Loss()  # ... loss on center point offset

        if opt.wh_loss == 'L1':
            self.crit_wh = L1Loss()  # ... loss on bounding box regression
        elif 'IoU' in opt.wh_loss:
            self.crit_wh = IoULoss(opt)  # ...opt decides which type of IoU loss
        else:
            raise AttributeError(f'Invalid argument for wh_loss: {opt.wh_loss}')
        
        self.names = ("total_loss", "cpt_hm_loss", "cpt_off_loss", "wh_loss")

    def forward(self, output, batch):
        """
        Combine the different loss terms to a total loss for a
        center point detection task. Loss term weights from Center Net Paper.

        Parameters
        ----------
        output: dictionary with output tensors
            network output of shape b x s x h x w,
            where s can be the number of classes or 2 for offset or
            bounding box regression
        batch: dictionary with target tensors
            shapes are either b x s x h x w (heat map) or
            b x n x 2 (offset or bounding box regression)

        Returns
        -------
        tuple of total loss (weighted sum of individual terms) and
        a dict of "loss_term_name": "loss_term_value" pairs to track
        the individual contribution of each term
        """
        total_loss = 0
        # apply activation function
        cpt_hm = _sigmoid(output["cpt_hm"])

        # calculate the loss on the center point heat map
        cpt_hm_loss = self.crit_hm(cpt_hm, batch["cpt_hm"])
        total_loss += cpt_hm_loss

        # calculate the loss of the center point offsets
        cpt_off_loss = self.crit_off(output["cpt_off"], batch["cpt_off"],
            batch["cpt_ind"], batch["cpt_mask"])
        total_loss += cpt_off_loss

        # calculate the loss on the bounding box dimensions
        if self.opt.normalize_wh:
            # here targets will also be in range [0, 1] thus
            wh = torch.sigmoid(output["wh"])  # to [0, 1]
            alpha = 20
        else:
            wh = output["wh"]
            alpha = 0.1

        if 'IoU' in self.opt.wh_loss:
            alpha = 10

        # wh loss will be displayed in tensorboard at its real
        # contribution to the total loss with its scale factor applied!
        wh_loss = self.crit_wh(wh, batch["wh"],
            batch["cpt_ind"], batch["cpt_mask"]) * alpha
        total_loss += wh_loss

        # keep track of individual loss terms
        loss_stats = {
            "cpt_hm_loss": cpt_hm_loss, "cpt_off_loss": cpt_off_loss,
            "wh_loss": wh_loss, "total_loss": total_loss
        }

        # only store each loss tensor's value!
        loss_stats = {k: v.item() for k, v in loss_stats.items()}
        # total_loss is still a tensor, needed for backward computation!
        return total_loss, loss_stats
