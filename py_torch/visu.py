import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np


def render(image, detections, opt, show=True, save=False, path=None):
    """
    Render the given image with bounding boxes and optionally class ids.
    Use batch size of 1.

    - image: 1 x 3 x h x w
    - detections: 1 x k x 6
    """
    image = image.detach().cpu()
    detections = detections.detach().cpu()
    fig, axs = plt.subplots()
    axs.set_axis_off()
    # denormalize image
    image = np.clip(255 * (opt.std * image[0].permute(1, 2, 0).numpy() + opt.mean), 0, 255).astype(np.uint8)
    axs.imshow(image, origin='upper')
    for det in detections[0]:
        bbox = det[:4]
        score = det[4]
        cid = int(det[5])
        rect = patches.Rectangle(bbox[:2], bbox[2], bbox[3],
                                 linewidth=2, edgecolor='r', facecolor='none')
        axs.add_patch(rect)
        text = axs.text(bbox[0] + 10, bbox[1] + 10, f"{cid:02d}; {score:.3f}",
                        fontsize=14, color="white")
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='black'),
            path_effects.Normal()
        ])

    if save:
        fig.savefig(path)
    if show:
        plt.show()

    plt.close(fig)
