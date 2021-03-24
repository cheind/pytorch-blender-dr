import torch
import numpy as np
import math

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
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
                # return torch.nan_to_num(iou - (rho2 / c2 + v * alpha), nan=1.0)  # CIoU
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

if __name__ == '__main__':
    # ground truth
    box1 = torch.tensor([150, 120, 50, 30])  # xmin, ymin, widht, height
    # detections
    box2 = torch.tensor([
        [150, 120, 50, 30],  # perfect match
        [150, 120, 30, 50],
        [140, 130, 50, 30],
        [10, 20, 50, 30],  # non overlapping
        [0, 0, 0, 0],  # invalid
    ])

    iou = bbox_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False)
    print('IoU:', iou, '==> bbox loss:', (1.0 - iou).mean())
    iou = bbox_iou(box1, box2, x1y1x2y2=False, GIoU=True, DIoU=False, CIoU=False)
    print('GIoU:', iou, '==> bbox loss:', (1.0 - iou).mean())
    iou = bbox_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=True, CIoU=False)
    print('DIoU:', iou, '==> bbox loss:', (1.0 - iou).mean())
    iou = bbox_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=True)
    print('CIoU:', iou, '==> bbox loss:', (1.0 - iou).mean())

    # special case checking
    box1 = torch.tensor([0, 0, 0, 0])  # xmin, ymin, widht, height
    box2 = torch.tensor([[0, 0, 0, 0]])  # xmin, ymin, widht, height
    iou = bbox_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False)
    print('IoU:', iou, '==> bbox loss:', (1.0 - iou).mean())
    iou = bbox_iou(box1, box2, x1y1x2y2=False, GIoU=True, DIoU=False, CIoU=False)
    print('GIoU:', iou, '==> bbox loss:', (1.0 - iou).mean())
    iou = bbox_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=True, CIoU=False)
    print('DIoU:', iou, '==> bbox loss:', (1.0 - iou).mean())
    iou = bbox_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=True)
    print('CIoU:', iou, '==> bbox loss:', (1.0 - iou).mean())
    