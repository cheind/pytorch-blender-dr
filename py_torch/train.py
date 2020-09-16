import torch
from torchvision.utils import make_grid

from .utils import MetricMeter
from .visu import render
from .decode import decode, filter_dets


def add_dt(writer, tag, n_iter, output, batch, opt):
    """ Add predicted image scene to tensorboard writer """
    dets = decode(output, opt.k)  # 1 x k x 6
    dets = filter_dets(dets, opt.model_thres)  # 1 x k' x 6

    image = batch["image"]  # original image
    dets[..., :4] = dets[..., :4] * opt.down_ratio

    fig = render(image, dets, opt, show=False, save=False, path=None, ret=True)
    writer.add_figure(tag, fig, global_step=n_iter, close=True)


def add_gt(writer, tag, n_iter, output, batch, opt):
    """ Add GT image scene to tensorboard writer """
    image = batch["image"]  # original image

    # build gt dets
    inds = batch["cpt_ind"]  # 1 x n_max
    wh = batch["wh"]  # 1 x n_max x 2
    cids = batch["cls_id"]  # 1 x n_max
    mask = batch["cpt_mask"].squeeze(0)  # n_max,
    
    ws = wh[..., 0]  # 1 x n_max
    hs = wh[..., 1]  # 1 x n_max
    wl = opt.w / opt.down_ratio
    ys = torch.true_divide(inds, wl).int().float()  # 1 x n_max
    xs = (inds % wl).int().float()  # 1 x n_max
    scores = torch.ones_like(cids)  # 1 x n_max
    
    dets = torch.stack([xs - ws / 2, ys - hs / 2, 
        ws, hs, scores, cids], dim=-1)  # 1 x n_max x 6

    dets = dets[:, mask.bool()]  # 1 x n' x 6
    dets[..., :4] = dets[..., :4] * opt.down_ratio

    fig = render(image, dets, opt, show=False, save=False, path=None, ret=True)
    writer.add_figure(tag, fig, global_step=n_iter, close=True)


def add_hms(writer, tag, n_iter, output, batch):
    # bring the output heat map in range to [0, 1]
    hm = [torch.sigmoid(output["cpt_hm"]), batch["cpt_hm"]]  # 1 x num_classes x hl x wl
    hm = [x.max(dim=1, keepdims=True)[0] for x in hm]  # 1 x 1 x hl x wl
    hm = torch.cat(hm, dim=0)  # 2 x 1 x hl x wl

    # visualize heat maps side by side
    hm = make_grid(hm, normalize=True, range=(0, 1), 
        pad_value=1)  # 3 x h x 2 * w + padding

    writer.add_image(tag, hm, n_iter)


def train(epoch, model, optimizer, dataloader, device, loss_fn, writer, opt):
    model.to(device=device)
    model.train()

    meter = MetricMeter()

    tag = "Train"

    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device=device) for k, v in batch.items()}

        output = model(batch["image"])
        loss, loss_dict = loss_fn(output, batch)

        meter.update(loss_dict)
        meter.to_writer(writer, tag, n_iter=(epoch - 1) * len(dataloader) + i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % opt.train_vis_interval == 0:
            batch = {k: v[:1].detach().clone().cpu() for k, v in batch.items()}
            output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}
            
            add_dt(writer, tag + "/DT", i, output, batch, opt)
            add_gt(writer, tag + "/GT", i, output, batch, opt)
            add_hms(writer, tag + "/HMS", i, output, batch)

    return meter


@torch.no_grad()
def eval(epoch, model, dataloader, device, loss_fn, writer, opt):
    model.to(device=device)
    model.train()

    meter = MetricMeter()

    tag = "Val"

    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device=device) for k, v in batch.items()}

        output = model(batch["image"])
        _, loss_dict = loss_fn(output, batch)

        meter.update(loss_dict)
        meter.to_writer(writer, tag, n_iter=(epoch - 1) * len(dataloader) + i)

        if i % opt.val_vis_interval == 0:
            batch = {k: v[:1].detach().clone().cpu() for k, v in batch.items()}
            output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}
            
            add_dt(writer, tag + "/DT", i, output, batch, opt)
            add_gt(writer, tag + "/GT", i, output, batch, opt)
            add_hms(writer, tag + "/HMS", i, output, batch)

    return meter
