import torch
from torchvision.utils import make_grid
from tqdm import tqdm

from .utils import MetricMeter
from .visu import render, render_class_distribution
from .decode import decode, filter_dets
from .coco_report import coco_eval, ap_values, draw_roc

def add_dt(writer, tag, n_iter, output, batch, opt):
    dets = decode(output, opt.k)  # 1 x k x 6
    dets = filter_dets(dets, opt.model_score_threshold)  # 1 x k' x 6

    image = batch["image"]  # original image
    dets[..., :4] = dets[..., :4] * opt.down_ratio

    fig = render(image, dets, opt, show=False, save=False, path=None, ret=True)
    writer.add_figure(tag, fig, global_step=n_iter, close=True)

def add_gt(writer, tag, n_iter, output, batch, opt):
    image = batch["image"]  # original image

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

    hm = make_grid(hm, normalize=True, range=(0, 1), 
        pad_value=1)  # 3 x h x 2 * w + padding

    writer.add_image(tag, hm, n_iter)

def add_cdistr(writer, tag, n_iter, cdistr, opt):
    """ Add class distribution bar plot to tensorboard writer """
    fig = render_class_distribution(cdistr, opt)
    writer.add_figure(tag, fig, global_step=n_iter, close=True)

def add_pr_curve(writer, tag, n_iter, prec, cids):
    """ Add precision/recall line plot to tensorboard writer """
    fig, ax = plt.subplots()
    draw_roc(prec, cids, ax)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    writer.add_figure(tag, fig, global_step=n_iter, close=True)

def use_multiple_devices(model):
    if torch.cuda.device_count() > 1 and torch.cuda.is_available():  
        num_devices = torch.cuda.device_count()
        logging.info(f"Available: {num_devices} GPUs!")
        assert opt.batch_size % num_devices == 0, 'Batch size not divisible by #GPUs'
        model = nn.DataParallel(model)  # Use all available devices
    
def save_checkpoint(save_dict, count, opt, stream=False):
    save_dict['batch' if stream else 'epoch'] = count

    torch.save(save_dict, 
        f"{opt.model_folder}/{opt.model_last_tag}.pth")

    if count % opt.save_interval == 0 and count != 0:
        torch.save(save_dict, 
            f"{opt.model_folder}/{opt.model_tag}_{count}.pth")

k

def train(epoch, model, optimizer, dataloader, loss_fn, writer, opt):
    device = next(model.parameters()).device
    model.train()
    meter = MetricMeter()

    with tqdm(total=len(dataloader)) as pbar:
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(batch["image"])
            loss, loss_dict = loss_fn(output, batch)

            meter.update(loss_dict)
            meter.to_writer(writer, "Train", n_iter=epoch * len(dataloader) + i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % opt.train_vis_interval == 0:
                batch = {k: v[:1].detach().clone().cpu() for k, v in batch.items()}
                output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}
                
                add_dt(writer, "Train/DT", i, output, batch, opt)
                add_gt(writer, "Train/GT", i, output, batch, opt)
                add_hms(writer, "Train/HMS", i, output, batch)
            pbar.set_postfix(loss=loss.item())
            pbar.update()
        pbar.close()

    return meter

@torch.no_grad()
def eval(epoch, model, dataloader, loss_fn, writer, opt):
    device = next(model.parameters()).device
    model.eval()
    meter = MetricMeter()

    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        output = model(batch["image"])
        _, loss_dict = loss_fn(output, batch)

        meter.update(loss_dict)
        meter.to_writer(writer, "Val", n_iter=epoch * len(dataloader) + i)

        if i % opt.val_vis_interval == 0:
            batch = {k: v[:1].detach().clone().cpu() for k, v in batch.items()}
            output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}
            
            add_dt(writer, "Val/DT", i, output, batch, opt)
            add_gt(writer, "Val/GT", i, output, batch, opt)
            add_hms(writer, "Val/HMS", i, output, batch)

    return meter

def stream_train(meter, batch_count, model, optimizer, 
    dataloader, loss_fn, writer, opt):
    device = next(model.parameters()).device
    model.train()

    with tqdm(total=opt.val_interval) as pbar:
        for i, batch in enumerate(dataloader):
            if i == opt.val_interval:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(batch["image"])
            loss, loss_dict = loss_fn(output, batch)

            meter.update(loss_dict)
            meter.to_writer(writer, "Train", n_iter=batch_count + i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % opt.train_vis_interval == 0:
                batch = {k: v[:1].detach().clone().cpu() for k, v in batch.items()}
                output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}
                
                add_dt(writer, "Train/DT", i, output, batch, opt)
                add_gt(writer, "Train/GT", i, output, batch, opt)
                add_hms(writer, "Train/HMS", i, output, batch)
            pbar.set_postfix(loss=loss.item())
            pbar.update()
        pbar.close()
    
    return batch_count + opt.val_interval

@torch.no_grad()
def stream_eval(meter, batch_count, model,  
    dataloader, loss_fn, writer, opt):
    device = next(model.parameters()).device
    model.eval()

    for i, batch in enumerate(dataloader):
        if i == opt.val_len:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        output = model(batch["image"])
        _, loss_dict = loss_fn(output, batch)

        meter.update(loss_dict)
        meter.to_writer(writer, "Val", n_iter=batch_count + i)

        if i % opt.val_vis_interval == 0:
            batch = {k: v[:1].detach().clone().cpu() for k, v in batch.items()}
            output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}
            
            add_dt(writer, "Val/DT", i, output, batch, opt)
            add_gt(writer, "Val/GT", i, output, batch, opt)
            add_hms(writer, "Val/HMS", i, output, batch)
    
    return batch_count + opt.val_len
