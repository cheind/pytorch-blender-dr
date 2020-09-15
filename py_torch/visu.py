import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np
import logging
import io


# define rgb colors for bounding boxes
# https://www.rapidtables.com/web/color/RGB_Color.html
COLORS = [
    [255, 102, 102],
    [255, 178, 102],
    [102, 255, 102],
    [102, 255, 255],
    [102, 102, 255],
    [255, 102, 255],
    [0, 102, 102],
    [102, 51, 0],
    [76, 0, 153],
    [0, 0, 0],
    [128, 128, 128],
]

COLORS = np.array(COLORS, dtype=np.float32) / 255


def image_from_figure(fig, close=True, dpi=96):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=dpi)
    io_buf.seek(0)
    # h x w x 4
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    if close:
        plt.close(fig)

    return img_arr[..., :3]  # h x w x 3


def tless_visu(item: dict, step, thres=0.3):
    """
    item = {
        "image": image,  # np.ndarray, h x w x 3
        "bboxes": self.all_bboxes[index],
        "cids": self.all_clsids[index],
        "bboxes_visib": self.all_bboxes_visib[index],
        "visib_fracts": self.all_visib_fracts[index]
    }
    """
    image = item["image"]
    bboxes = item["bboxes"]
    cids = item["cids"]
    bboxes_visib = item["bboxes_visib"]
    visib_fracts = item["visib_fracts"]

    h, w = image.shape[:2]
    DPI = 96
    fig = plt.figure(frameon=False, figsize=(2 * w/DPI, h/DPI), dpi=DPI)
    axs = [fig.add_axes([0,0,0.5,0.5]), fig.add_axes([0.5,0.0,0.5,0.5])]

    axs[0].imshow(image, origin='upper')
    for bbox, cid, visib_fract in zip(bboxes, cids, visib_fracts):
        col = COLORS[int(cid)]
        rect = patches.Rectangle(bbox[:2], bbox[2], bbox[3],
                                 linewidth=2, edgecolor=col, 
                                 facecolor='none')
        rect.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground="white"),
            path_effects.Normal()
        ])
        axs[0].add_patch(rect)
        text = axs[0].text(bbox[0] + 10, bbox[1] + 10, f"{cid} {visib_fract:.3f}",
                        fontsize=12, color=col)
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground="white"),
            path_effects.Normal()
        ])
    
    axs[1].imshow(image, origin='upper')
    for bbox, cid, visib_fract in zip(bboxes_visib, cids, visib_fracts):
        if visib_fract < thres:
            continue
        col = COLORS[int(cid)]
        rect = patches.Rectangle(bbox[:2], bbox[2], bbox[3],
                                 linewidth=2, edgecolor=col, 
                                 facecolor='none')
        rect.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground="white"),
            path_effects.Normal()
        ])
        axs[1].add_patch(rect)
        text = axs[1].text(bbox[0] + 10, bbox[1] + 10, f"{cid} {visib_fract:.3f}",
                        fontsize=12, color=col)
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground="white"),
            path_effects.Normal()
        ])

    for i in range(len(axs)):
        axs[i].set_axis_off()
        axs[i].set_xlim(0,w-1)
        axs[i].set_ylim(h-1,0)

    fn = f'./debug/visu_{step}.png'
    fig.savefig(fn)
    logging.info(f"Saved image at: {fn}")
    plt.close(fig)


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
                        #f"{score:.3f}",
                        fontsize=12, color=col)
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground="white"),
            path_effects.Normal()
        ])

    if save:
        fig.savefig(path)
        logging.info(f"Figure saved under: {path}")
    if show:
        plt.show()
    if ret:
        return fig
    else:
        plt.close(fig)
