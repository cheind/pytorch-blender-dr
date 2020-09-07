from contextlib import ExitStack
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
import numpy as np
import logging
import os

import torch
from torch import optim
from torch.utils import data

from blendtorch import btt

from .utils import Config
from .train import train, eval
from .loss import CenterLoss
from .model import get_model
from .visu import render


class Transformation:

    def __init__(self, opt):
        self.h, self.w = opt.h, opt.w  # e.g. 512 x 512
        self.num_classes = opt.num_classes  # num. of object classes
        self.n_max = opt.n_max  # num. of max objects per image
        self.down_ratio = 4  # => low resolution 128 x 128
        # ImageNet stats
        self.mean = opt.mean
        self.std = opt.std

        transformations = [
            A.ChannelShuffle(p=0.5),
            A.HorizontalFlip(p=0.2),
        ] if opt.augment else []

        transformations.extend([
            # Rescale an image so that minimum side is equal to max_size,
            # keeping the aspect ratio of the initial image.
            A.SmallestMaxSize(max_size=min(opt.h, opt.w)),
            A.RandomCrop(height=opt.h, width=opt.w) if opt.augment else A.CenterCrop(height=opt.h, width=opt.w),
            A.Normalize(mean=opt.mean, std=opt.std),
        ])

        bbox_params = A.BboxParams(
            format="coco",
            min_area=100,  # < 100 pixels => drop bbox
            min_visibility=0.2,  # < 20% of orig. vis. => drop bbox
        )

        self.transform_fn = A.Compose(transformations, bbox_params=bbox_params)

    def gen_map(self, shape, xy: np.ndarray,  # cids: np.ndarray, num_classes,
                mask=None, sigma=4, cutoff=1e-3, normalize=False, bleed=True):
        """
        Generates a single belief map of 'shape' for each point in 'xy'.

        Parameters
        ----------
        shape: tuple
            h x w of image
        xy: n x 2
            n points with x, y coordinates (image coordinate system)
        cids: n,
            class ids
        num_classes: scalar
            num. of classes
        mask: n,
            zero-one mask to select points from xy
        sigma: scalar
            gaussian sigma
        cutoff: scalar
            set belief to zero if it is less then cutoff
        normalize: bool
            whether to multiply with the gaussian normalization factor or not

        Returns
        -------
        belief map: 1 x h x w  # num_classes x h x w
        """
        n = xy.shape[0]
        h, w = shape[:2]

        if n == 0:
            return np.zeros((1, h, w), dtype=np.float32)  # np.zeros((num_classes, h, w), dtype=np.float32)

        if not bleed:
            wh = np.asarray([w - 1, h - 1])[None, :]
            mask_ = np.logical_or(xy[..., :2] < 0, xy[..., :2] > wh).any(-1)
            xy = xy.copy()
            xy[mask_] = np.nan

        # grid is 2 x h x h
        grid = np.array(np.meshgrid(np.arange(w), np.arange(h)), dtype=np.float32)
        # reshape grid to 1 x 2 x h x w
        grid = grid.reshape((1, 2, h, w))
        # reshape xy to n x 2 x 1 x 1
        xy = xy.reshape((n, 2, 1, 1))
        # compute squared distances to joints
        d = ((grid - xy) ** 2).sum(1)
        # compute gaussian
        b = np.nan_to_num(np.exp(-(d / (2.0 * sigma ** 2))))

        if normalize:
            b = b / np.sqrt(2 * np.pi) / sigma  # n x h x w

        # b is n x h x w
        b[(b < cutoff)] = 0

        if mask is not None:
            # set the invalid center point maps to all zero
            b *= mask[:, None, None]  # n x h x w

        return b.max(0, keepdims=True)  # 1 x h x w

    def item_transform(self, item):
        """
        Transform data for training.

        :param item: dictionary
            - image: h x w x 3
            - bboxes: n x 4; [[x, y, width, height], ...]
            - cids: n,

        :return: dictionary
            - image: 3 x 512 x 512
            - cpt_hm: 1 x 128 x 128 # num_classes x 128 x 128
            - cpt_off: n_max x 2 low resolution offset - [0, 1)
            - cpt_ind: n_max, low resolution indices - [0, 128^2)
            - cpt_mask: n_max,
            - wh: n_max x 2, low resolution width, height - [0, 128], [0, 128]
            - cls_id: n_max,
        """
        image = item["image"]
        bboxes = item['bboxes']
        cids = item["cids"]

        h, w = image.shape[:2]

        # bboxes = np.array([[500, 400, 40, 12]], dtype=np.float32).repeat(9, 0)
        # bboxes = np.concatenate((bboxes, np.array([[200, 200, 50, 50]])))

        # adjust bboxes to pass initial - check_bbox(bbox) - call
        # from the albumentations package
        # library can't deal with bbox corners outside of the image
        x, y, bw, bh = np.split(bboxes, 4, -1)  # each n x 1
        x[x < 0] = 0
        y[y < 0] = 0
        x[x > w] = w
        y[y > h] = h
        bw[x + bw > w] = w - x[x + bw > w]
        bh[y + bh > h] = h - y[y + bh > h]
        bboxes = np.concatenate((x, y, bw, bh), axis=-1)
        # note: further processing is done by albumentations, bboxes
        # are dropped when not satisfying min. area or visibility!

        # prepare bboxes for transformation
        bbox_labels = np.arange(len(bboxes), dtype=np.float32)
        bboxes = np.append(bboxes, bbox_labels[:, None], axis=-1)

        transformed = self.transform_fn(image=image, bboxes=bboxes)
        image = np.array(transformed["image"], dtype=np.float32)
        image = image.transpose((2, 0, 1))  # 3 x h x w
        bboxes = np.array(transformed["bboxes"], dtype=np.float32)

        # bboxes can be dropped
        len_valid = len(bboxes)

        # to be batched we have to bring everything to the same shape
        cpt = np.zeros((self.n_max, 2), dtype=np.float32)
        # get center points of bboxes (image coordinates)
        cpt[:len_valid, 0] = bboxes[:, 0] + bboxes[:, 2] / 2  # x
        cpt[:len_valid, 1] = bboxes[:, 1] + bboxes[:, 3] / 2  # y

        cpt_mask = np.zeros((self.n_max,), dtype=np.uint8)
        cpt_mask[:len_valid] = 1

        wh = np.zeros((self.n_max, 2), dtype=np.float32)
        wh[:len_valid, :] = bboxes[:, 2:-1] / self.down_ratio

        cls_id = np.zeros((self.n_max,), dtype=np.uint8)
        # the bbox labels help to reassign the correct classes
        cls_id[:len_valid] = cids[bboxes[:, -1].astype(np.int32)]

        # LOW RESOLUTION dimensions
        hl, wl = int(self.h / self.down_ratio), int(self.w / self.down_ratio)
        cpt = cpt / self.down_ratio

        # discrete center point coordinates
        cpt_int = cpt.astype(np.int32)

        cpt_ind = np.zeros((self.n_max,), dtype=np.int64)
        # index = y * wl + x
        cpt_ind[:len_valid] = cpt_int[:len_valid, 1] * wl + cpt_int[:len_valid, 0]

        cpt_off = np.zeros((self.n_max, 2), dtype=np.float32)
        cpt_off[:len_valid] = (cpt - cpt_int)[:len_valid]

        cpt_hm = self.gen_map((hl, wl), cpt, mask=cpt_mask)  # 1 x hl x wl

        item = {
            "image": image,
            "cpt_hm": cpt_hm,
            "cpt_off": cpt_off,
            "cpt_ind": cpt_ind,
            "cpt_mask": cpt_mask,
            "wh": wh,
            "cls_id": cls_id
        }
        return item


def iterate(dl):
    DPI=96
    for step, item in enumerate(dl):
        img, bboxes, cids = item['image'], item['bboxes'], item['cids']
        H, W = img.shape[2:]  # img: b x 3 x h x w
        fig = plt.figure(frameon=False, figsize=(W*2/DPI,H*2/DPI), dpi=DPI)
        axs = [fig.add_axes([0,0,0.5,0.5]), fig.add_axes([0.5,0.0,0.5,0.5]), fig.add_axes([0.0,0.5,0.5,0.5]), fig.add_axes([0.5,0.5,0.5,0.5])]
        for i in range(img.shape[0]):
            axs[i].imshow(img[i].permute(1, 2, 0), origin='upper')
            for cid, bbox in zip(cids[i],bboxes[i]):
                rect = patches.Rectangle(bbox[:2],bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')
                axs[i].add_patch(rect)
                axs[i].text(bbox[0]+10, bbox[1]+10, f'Class {cid.item()}', fontsize=18)
            axs[i].set_axis_off()
            axs[i].set_xlim(0,W-1)
            axs[i].set_ylim(H-1,0)
        fig.savefig(f'./data/output_{step}.png')
        plt.close(fig)


def main(opt):
    transformation = Transformation(opt)
    item_transform = transformation.item_transform

    with ExitStack() as es:
        if not opt.replay:
            # Launch Blender instance. Upon exit of this script all Blender instances will be closed.
            bl = es.enter_context(
                btt.BlenderLauncher(
                    scene=f"{opt.scene}.blend",
                    script=f"{opt.scene}.blend.py",
                    num_instances=opt.blender_instances,
                    named_sockets=['DATA'],
                    blend_path=opt.blend_path,
                )
            )

            # Setup a streaming dataset
            ds = btt.RemoteIterableDataset(
                bl.launch_info.addresses['DATA'],
                item_transform=item_transform
            )
            # Iterable datasets do not support shuffle
            shuffle = False

            # Limit the total number of streamed elements
            ds.stream_length(4)

            # Setup raw recording if desired
            if opt.record:
                name = os.path.basename(opt.scene)
                ds.enable_recording(f'./data/record_{name}')
        else:
            # Otherwise we replay from file.
            name = os.path.basename(opt.scene)
            ds = btt.FileDataset(f'./data/record_{name}', item_transform=item_transform)
            shuffle = False

        # try to over fit on a single example
        ds = data.Subset(ds, indices=[0])

        # Setup DataLoader and iterate
        dl = data.DataLoader(ds, batch_size=opt.batch_size, num_workers=opt.worker_instances, shuffle=shuffle)

        # batch = next(iter(dl))
        # for k, v in batch.items():
        #     print(k, v.shape)
        #
        # plt.imshow(batch["cpt_hm"].squeeze(0).permute(1, 2, 0).numpy())
        # plt.show()
        # plt.imshow(batch["image"].squeeze(0).permute(1, 2, 0).numpy())
        # plt.show()

        if opt.record:
            print("Generating images of the recorded data...")
            iterate(dl)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # heads - head_name: num. channels of model
        heads = {"cpt_hm": opt.num_classes, "cpt_off": 2, "wh": 2}
        model = get_model(heads)

        # t = torch.randn(4, 3, 512, 512)
        # out = model(t)
        # print(out["cpt_hm"].shape,
        #       out["cpt_off"].shape,
        #       out["wh"].shape)
        #
        #
        # output, batch

        loss_fn = CenterLoss()

        # batch = next(iter(dl))
        # out = model(batch["image"])
        # test_loss = loss_fn(out, batch)

        optimizer = optim.Adam(model.parameters(), opt.lr)

        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter()  # save into ./runs folder

        # tensorboard --logdir=runs

        for epoch in range(1, opt.num_epochs + 1):
            logging.info(f"Inside trainings loop at epoch: {epoch}")
            train(epoch, model, optimizer, dl, device, loss_fn, writer)

            if epoch % opt.val_interval == 0:
                eval(epoch, model, dl, device, loss_fn, writer)

        PATH = "./models/center_net.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, PATH)

        checkpoint = torch.load(PATH)
        heads = {"cpt_hm": opt.num_classes, "cpt_off": 2, "wh": 2}
        model = get_model(heads)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), opt.lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        model.eval()

        batch = next(iter(dl))
        out = model(batch["image"])
        test_loss = loss_fn(out, batch)
        print(test_loss)


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)

    opt = Config("./configs/config.txt")
    print(opt)

    main(opt)
