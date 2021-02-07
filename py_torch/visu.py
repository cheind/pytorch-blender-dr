import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np
import logging
import io
import cv2

from .constants import MAPPING, COLORS
from .evaluation import _to_float

def render(image, detections, opt, show=True, 
    save=False, path=None, denormalize=True, ret=False):
    """
    Render the given image with bounding boxes and optionally class ids.
    Use batch size of 1.

    - image: 1 x 3 x h x w torch.Tensor or 1 x h x w x 3 np.ndarray
    - detections: 1 x k x 6
    """
    image = image.detach().clone().cpu()
    if denormalize:  # denormalize image and reshape to h x w x 3
        image = np.clip(255 * (opt.std * image[0].permute(1, 2, 0).numpy() + opt.mean), 0, 255).astype(np.uint8)
    else:  # ground truth image is 1 x h x w x 3 and not normalized
        image = image[0]  # batch size of 1

    h, w = image.shape[:2]

    detections = detections.detach().clone().cpu()
    DPI = 96
    fig = plt.figure(frameon=False, figsize=(w/DPI, h/DPI), dpi=DPI)
    axs = fig.add_axes([0, 0, 1.0, 1.0])
    axs.set_axis_off()
    
    axs.imshow(image, origin='upper')
    for i, det in enumerate(detections[0]):
        bbox = det[:4]
        score = det[4]
        cid = int(det[5])
        col = COLORS[cid]
        rect = patches.Rectangle(bbox[:2], bbox[2], bbox[3],
                                 linewidth=2, edgecolor=col, 
                                 facecolor='none')
        rect.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground="white"),
            path_effects.Normal()
        ])
        axs.add_patch(rect)
        text = axs.text(bbox[0] + 10, bbox[1] + 10, f"{cid} {score:.3f}",
                        fontsize=12, color=col)
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground="white"),
            path_effects.Normal()
        ])

    if save:
        fig.savefig(path)
    if show:
        plt.show()
    if ret:
        return fig
    else:
        plt.close(fig)

def render_class_distribution(cdistr: dict, opt):
    """
    Create bar plot of class distribution.
    
    - cls_distr: keys (=labels) from 0 - 29

    note: original 1-30 but inside Blender we took
    more natural indices from 0-29, we want to see the
    actual class distribution with classes 0-5 used
    in our experiments 
    """
    for old_cid in range(1, 31):  # 1 - 30
        new_cid = MAPPING[old_cid]  # 0 - 5
        cd[new_cid] += cdistr[old_cid - 1]  # 0 - 29
    
    # initialize bar plot
    fig, ax = plt.subplots()
    cd = [0 for _ in range(opt.num_classes)]  
    bars = ax.bar(x=list(range(opt.num_classes)), 
        height=cd, color=COLORS[:opt.num_classes])

    total = sum(cd)

    def autolabel(bars):
        """
        Attach a text label above each bar in bars, 
        displaying its height.
        """
        for bar in barss:
            height = bar.get_height()
            ax.annotate(f'{_to_float(height)} / {int(height / total * 100)}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(bars)
    fig.tight_layout()
    fig.add_axes(ax)
    return fig

def iterate(dl, opt):
    DPI=96
    for step, item in enumerate(dl):
        img, bboxes, cids = item['image'], item['bboxes'], item['cids']
        H, W = img.shape[2:]  # img: b x 3 x h x w
        fig = plt.figure(frameon=False, figsize=(W*2/DPI,H*2/DPI), dpi=DPI)
        axs = [fig.add_axes([0,0,0.5,0.5]), fig.add_axes([0.5,0.0,0.5,0.5]), 
            fig.add_axes([0.0,0.5,0.5,0.5]), fig.add_axes([0.5,0.5,0.5,0.5])]
        for i in range(img.shape[0]):
            axs[i].imshow(img[i].permute(1, 2, 0), origin='upper')
            for cid, bbox in zip(cids[i],bboxes[i]):
                rect = patches.Rectangle(bbox[:2], bbox[2], bbox[3], 
                    linewidth=2, edgecolor='r', facecolor='none')
                axs[i].add_patch(rect)
                axs[i].text(bbox[0]+10, bbox[1]+10, f'Class {cid.item()}', fontsize=18)
            axs[i].set_axis_off()
            axs[i].set_xlim(0,W-1)
            axs[i].set_ylim(H-1,0)
        fig.savefig(f'{opt.debug_path}/{step}.png')
        plt.close(fig)

def check_batch_from_loader(dl, opt):
    DPI = 96
    batch = next(iter(dl))      
    for j in range(opt.batch_size):
        batch_ = {k: v[j:j+1, ...] for k, v in batch.items()}
        image = batch_["image"]  # 3 x h x w
        inds = batch_["cpt_ind"]  # 1 x n_max
        wh = batch_["wh"]  # 1 x n_max x 2
        cids = batch_["cids"]  # 1 x n_max
        mask = batch_["cpt_mask"].squeeze(0)  # n_max,
        
        wl = opt.w / opt.down_ratio
        ys = torch.true_divide(inds, wl).int().float()  # 1 x n_max
        xs = (inds % wl).int().float()  # 1 x n_max

        scores = torch.ones_like(cids)  # 1 x n_max

        ws = wh[..., 0]  # 1 x n_max
        hs = wh[..., 1]  # 1 x n_max
        dets = torch.stack([xs - ws / 2, ys - hs / 2, 
            ws, hs, scores, cids], dim=-1)  # 1 x n_max x 6
        dets = dets[:, mask.bool()]  # 1 x n' x 6
        dets[..., :4] = dets[..., :4] * opt.down_ratio

        render(image, dets, opt, show=False, save=True, 
            denormalize=True, path=f"{opt.debug_path}/{j:03d}_det.png", 
            ret=False)
        
        hm = torch.sigmoid(batch_["cpt_hm"])  # 1 x num_classes x hl x wl
        hm = hm.max(dim=1, keepdims=False)[0]  # hl x wl
        assert hm.ndim == 2
        hm = cv2.resize(hm, dsize=(opt.w, opt.h))
        # increase spatial dimensions to match detection image size
        fig = plt.figure(frameon=False, figsize=(opt.w/DPI, opt.h/DPI), dpi=DPI)
        plt.imshow(hm, cmap='gray')
        fig.savefig(f"{opt.debug_path}/{j:03d}_hm.png")
