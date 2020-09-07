import torch

from .utils import MetricMeter


def train(epoch, model, optimizer, dataloader, device, loss_fn, writer):
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

    return meter


@torch.no_grad()
def eval(epoch, model, dataloader, device, loss_fn, writer):
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

    return meter
