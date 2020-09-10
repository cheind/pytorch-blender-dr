import torch

from .utils import MetricMeter
from .visu import image_from_figure, render
from .decode import decode, filter_dets


def to_image(output, batch, opt):
    """ visualize one image from the batch """
    output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}  # batch size of 1
    dets = decode(output, opt.k)  # 1 x k x 6
    dets = filter_dets(dets, opt.thres)

    image = batch["image"][:1].detach().clone().cpu()  # original image
    dets[..., :4] = dets[..., :4] * opt.down_ratio

    fig = render(image, dets, opt, show=False, save=False, 
        path=None, ret=True)
    image = image_from_figure(fig, close=True)

    return image.transpose(2, 0, 1)  # 3 x h x w 


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
            image = to_image(output, batch, opt)
            writer.add_image("Images/Train", image, i)

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
            image = to_image(output, batch, opt)
            writer.add_image("Images/Val", image, i)

    return meter
