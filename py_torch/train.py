import torch

from .utils import MetricMeter


def train(epoch, model, optimizer, dataloader, device, loss_fn):
    model.to(device=device)
    model.train()

    meter = MetricMeter()

    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device=device) for k, v in batch.items()}

        output = model(batch["image"])
        loss, loss_dict = loss_fn(output, batch)

        meter.update(loss_dict)
        print("Epoch:", epoch, str(meter))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def eval(epoch, model, dataloader, device, loss_fn):
    model.to(device=device)
    model.train()

    meter = MetricMeter()

    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device=device) for k, v in batch.items()}

        output = model(batch["image"])
        _, loss_dict = loss_fn(output, batch)

        meter.update(loss_dict)
        print("Epoch:", epoch, str(meter))
