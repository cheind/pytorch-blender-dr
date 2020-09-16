import torch
from torchvision.utils import make_grid

from .utils import MetricMeter
from .visu import image_from_figure, render
from .decode import decode, filter_dets


def to_image(output, batch, opt):
    """ Visualize one image from the batch. """
    output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}  # batch size of 1
    dets = decode(output, opt.k)  # 1 x k x 6
    dets = filter_dets(dets, opt.model_thres)  # 1 x k' x 6

    image = batch["image"][:1].detach().clone().cpu()  # original image
    dets[..., :4] = dets[..., :4] * opt.down_ratio

    fig = render(image, dets, opt, show=False, save=False, 
        path=None, ret=True)
    image_pred = image_from_figure(fig, close=True).transpose(2, 0, 1)

    # build gt dets
    inds = batch["cpt_ind"][:1].detach().clone().cpu()  # 1 x n_max
    wh = batch["wh"][:1].detach().clone().cpu()  # 1 x n_max x 2
    cids = batch["cls_id"][:1].detach().clone().cpu()  # 1 x n_max
    mask = batch["cpt_mask"][0].detach().clone().cpu()  # n_max,
    
    ws = wh[..., 0]  # 1 x n_max
    hs = wh[..., 1]  # 1 x n_max
    ys = torch.true_divide(inds, 128).int().float()  # 1 x n_max
    xs = (inds % 128).int().float()  # 1 x n_max
    scores = torch.ones_like(cids)  # 1 x n_max
    
    dets = torch.stack([xs - ws / 2,
                        ys - hs / 2, 
                        ws, 
                        hs, 
                        scores, cids], dim=-1)  # 1 x n_max x 6

    dets = dets[:, mask.bool()]  # 1 x n' x 6

    # bboxes from 128 x 128 space to 512 x 512
    dets[..., :4] = dets[..., :4] * opt.down_ratio

    fig = render(image, dets, opt, show=False, save=True, 
        path="./data/test.png", ret=True)
    image_gt = image_from_figure(fig, close=True).transpose(2, 0, 1)

    # 1 x num_classes x hl x wl
    hm = [output["cpt_hm"][:1].detach().clone().cpu(),
        batch["cpt_hm"][:1].detach().clone().cpu()]
    
    print("min, max", torch.min(hm[0]), torch.max(hm[0]))
    hm = [torch.sigmoid(x) for x in hm]  # range 
    print("min, max", torch.min(hm[0]), torch.max(hm[0]))
    
    # 1 x 1 x hl x wl
    hm = [x.max(dim=1, keepdims=True)[0] for x in hm]
    
    hm = torch.cat(hm, dim=0)  # 2 x 1 x hl x wl

    hm = make_grid(hm, normalize=True, range=(0, 1), 
        pad_value=1)  # 3 x h x 2 * w + padding

    return image_pred, hm, image_gt  # 3 x h x w 


def train(epoch, model, optimizer, dataloader, device, loss_fn, writer, opt):
    model.to(device=device)
    model.train()

    meter = MetricMeter()

    tag = "Loss/Train"

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
            image_pred, hm, image_gt = to_image(output, batch, opt)
            writer.add_image("Images/Train", image_pred, i)
            writer.add_image("Images/Train GT", image_gt, i)
            writer.add_image("Images/Train Heat Map", hm, i)

    return meter


@torch.no_grad()
def eval(epoch, model, dataloader, device, loss_fn, writer, opt):
    model.to(device=device)
    model.train()

    meter = MetricMeter()

    tag = "Loss/Val"

    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device=device) for k, v in batch.items()}

        output = model(batch["image"])
        _, loss_dict = loss_fn(output, batch)

        meter.update(loss_dict)
        meter.to_writer(writer, tag, n_iter=(epoch - 1) * len(dataloader) + i)

        if i % opt.val_vis_interval == 0:
            image_pred, hm, image_gt = to_image(output, batch, opt)
            writer.add_image("Images/Val", image_pred, i)
            writer.add_image("Images/Val GT", image_gt, i)
            writer.add_image("Images/Val Heat Map", hm, i)

    return meter
