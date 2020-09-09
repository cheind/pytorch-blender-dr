import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np
import logging


def render(image, detections, opt, show=True, save=False, path=None, denormalize=True):
    """
    Render the given image with bounding boxes and optionally class ids.
    Use batch size of 1.

    - image: 1 x 3 x h x w torch.Tensor or 1 x h x w x 3 np.ndarray
    - detections: 1 x k x 6
    """
    image = image.detach().cpu()
    if denormalize:  # denormalize image and reshape to h x w x 3
        image = np.clip(255 * (opt.std * image[0].permute(1, 2, 0).numpy() + opt.mean), 0, 255).astype(np.uint8)
    else:  # ground truth image is 1 x h x w x 3 and not normalized
        image = image[0]  # batch size of 1

    h, w = image.shape[:2]

    detections = detections.detach().cpu()
    DPI = 96
    fig = plt.figure(frameon=False, figsize=(w/DPI, h/DPI), dpi=DPI)
    axs = fig.add_axes([0, 0, 1.0, 1.0])
    axs.set_axis_off()
    
    axs.imshow(image, origin='upper')
    for det in detections[0]:
        bbox = det[:4]
        score = det[4]
        # cid = int(det[5])
        rect = patches.Rectangle(bbox[:2], bbox[2], bbox[3],
                                 linewidth=2, edgecolor='r', facecolor='none')
        axs.add_patch(rect)
        text = axs.text(bbox[0] + 10, bbox[1] + 10, # f"{cid:02d}; {score:.3f}",
                        f"{score:.3f}",
                        fontsize=14, color="white")
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='black'),
            path_effects.Normal()
        ])

    if save:
        fig.savefig(path)
        logging.info(f"Figure saved under: {path}")
    if show:
        plt.show()

    plt.close(fig)
