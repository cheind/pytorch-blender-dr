import torch
from torchvision.utils import make_grid
from tqdm import tqdm
from pycocotools.coco import COCO
import json
import logging
import matplotlib.pyplot as plt
from torch import nn
from torch.cuda.amp import autocast

from .utils import MetricMeter
from .visu import render, render_class_distribution
from .decode import decode, filter_dets
from .coco_report import coco_eval, ap_values, draw_roc
from .evaluation import _to_float
from .constants import GROUPS

def add_dt(writer, tag, n_iter, output, batch, opt):
    dets = decode(output, opt.k)  # 1 x k x 6
    dets = filter_dets(dets, opt.model_score_threshold_high)  # 1 x k' x 6

    image = batch["image"]  # original image
    dets[..., :4] = dets[..., :4] * opt.down_ratio

    fig = render(image, dets, opt, show=False, save=False, path=None, ret=True)
    writer.add_figure(tag, fig, global_step=n_iter, close=True)

def add_gt(writer, tag, n_iter, output, batch, opt):
    image = batch["image"]  # original image

    inds = batch["cpt_ind"]  # 1 x n_max
    wh = batch["wh"]  # 1 x n_max x 2
    cids = batch["cids"]  # 1 x n_max
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
    hm = [torch.sigmoid(output["cpt_hm"].float()), batch["cpt_hm"]]  # 1 x classes x hl x wl
    hm = [x.max(dim=1, keepdims=True)[0] for x in hm]  # 1 x 1 x hl x wl
    hm = torch.cat(hm, dim=0)  # 2 x 1 x hl x wl

    hm = make_grid(hm, normalize=True, range=(0, 1), 
        pad_value=1)  # 3 x h x 2 * w + padding

    writer.add_image(tag, hm, n_iter)

def add_cdistr(writer, tag, n_iter, cdistr, opt):
    """ Add class distribution bar plot to tensorboard writer """
    fig = render_class_distribution(cdistr, opt)
    writer.add_figure(tag, fig, n_iter, close=True)

def add_pr_curve(writer, tag, n_iter, prec, cids):
    """ Add precision/recall line plot to tensorboard writer """
    fig, ax = plt.subplots()
    draw_roc(prec, cids, ax)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    writer.add_figure(tag, fig, n_iter, close=True)

def save_checkpoint(save_dict, count, metric, loss, best_metric, 
    best_loss, opt):

    torch.save(save_dict, 
        f"{opt.model_folder}/{opt.model_last_tag}.pth")

    if count % opt.save_interval == 0 and count != 0:
        torch.save(save_dict, 
            f"{opt.model_folder}/{opt.model_tag}_{count}.pth")
    
    if loss <= best_loss and opt.use_loss:
        best_loss = loss
        save_dict['best_loss'] = best_loss
        torch.save(save_dict, 
            f'{opt.model_folder}/{opt.best_model_tag}.pth')

    if metric >= best_metric and opt.use_metric:
        best_metric = metric
        save_dict['best_metric'] = best_metric
        torch.save(save_dict, 
            f'{opt.model_folder}/{opt.best_model_tag}.pth')
    
    return best_metric, best_loss

def train(meter, epoch, model, optimizer, loader, 
    loss_fn, writer, opt):
    device = next(model.parameters()).device
    model.train()
    meter.reset()

    with tqdm(total=len(loader)) as pbar:
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(batch["image"])
            loss, loss_dict = loss_fn(output, batch)

            meter.update(loss_dict)
            meter.to_writer(writer, "Train", n_iter=epoch * len(loader) + i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % opt.train_vis_interval == 0:
                batch = {k: v[:1].detach().clone().cpu() for k, v in batch.items()}
                output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}
                
                add_dt(writer, "Train/DT", epoch * len(loader) + i, output, batch, opt)
                add_gt(writer, "Train/GT", epoch * len(loader) + i, output, batch, opt)
                add_hms(writer, "Train/HMS", epoch * len(loader) + i, output, batch)
            pbar.set_postfix(loss=loss.item())
            pbar.update()
        pbar.close()
        
def train_new(meter, epoch, model, optimizer, scaler, loader, 
    loss_fn, writer, opt):
    device = next(model.parameters()).device
    model.train()
    meter.reset()
    
    pbar = enumerate(loader)
    if opt.rank == 0:
        pbar = tqdm(pbar, total=len(loader), desc='train')
        if opt.cuda:
            torch.cuda.reset_peak_memory_stats(device)
        
    for i, batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast(enabled=opt.cuda):
            output = model(batch["image"])
            loss, loss_dict = loss_fn(output, batch)
        
        if opt.rank == 0:
            # memory consumption is always highest after forward pass and 
            # before backward pass since activations have to be stored!
            mem = torch.cuda.memory_reserved() / 1E9 if opt.cuda else 0
            max_mem = torch.cuda.max_memory_reserved(device)  / 1E9 if opt.cuda else 0
            
        scaler.scale(loss).backward()  # accumulates scaled gradients
        if (i + 1) % opt.accumulate == 0:  # number of batches to accumulate
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if opt.rank == 0:
            meter.update(loss_dict)
            meter.to_writer(writer, "Train", n_iter=epoch * len(loader) + i)
            
            pf = f'mem: {mem:.2f}/{max_mem:.2f}GB, '
            pf += f'loss: {loss:.2f}, '  # total loss 
            pf += f'epoch: {epoch}/{opt.start_epoch + opt.epochs - 1}'
            pbar.set_postfix_str(pf)

            if i % opt.train_vis_interval == 0:
                batch = {k: v[:1].detach().clone().cpu() for k, v in batch.items()}
                output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}

                add_dt(writer, "Train/DT", epoch * len(loader) + i, output, batch, opt)
                add_gt(writer, "Train/GT", epoch * len(loader) + i, output, batch, opt)
                add_hms(writer, "Train/HMS", epoch * len(loader) + i, output, batch)
                
@torch.no_grad()
def val_new(meter, epoch, model, loader, loss_fn, writer, opt):
    device = next(model.parameters()).device
    model.eval()
    meter.reset()
    
    # initialize variables for metric calculation
    image_id = -1    
    gt_ann_id = -1
    ground_truth = empty_gt_dict()
    prediction = []

    pbar = enumerate(loader)
    pbar = tqdm(pbar, total=len(loader), desc='val')
    
    if opt.cuda:
        torch.cuda.reset_peak_memory_stats(device)
    
    for i, batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast(enabled=opt.cuda):
            output = model(batch["image"])
            loss, loss_dict = loss_fn(output, batch)
        
        mem = torch.cuda.memory_reserved() / 1E9 if opt.cuda else 0
        max_mem = torch.cuda.max_memory_reserved(device)  / 1E9 if opt.cuda else 0

        # evolve variables for metric calculation
        image_id, gt_ann_id = build_annotations(output, batch, 
            image_id, gt_ann_id, ground_truth, prediction, opt)
        
        meter.update(loss_dict)
        meter.to_writer(writer, "Val", n_iter=epoch * len(loader) + i)

        pf = f'mem: {mem:.2f}/{max_mem:.2f}GB, '
        pf += f'loss: {loss:.2f}, '  # total loss 
        pf += f'epoch: {epoch}/{opt.start_epoch + opt.epochs - 1}'
        pbar.set_postfix_str(pf)

        if i % opt.val_vis_interval == 0:
            batch = {k: v[:1].detach().clone().cpu() for k, v in batch.items()}
            output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}

            add_dt(writer, "Val/DT", epoch * len(loader) + i, output, batch, opt)
            add_gt(writer, "Val/GT", epoch * len(loader) + i, output, batch, opt)
            add_hms(writer, "Val/HMS", epoch * len(loader) + i, output, batch)

    try:
        # save, load and evaluate
        gtFile = f'{opt.evaluation_folder}/gt.json'
        dtFile = f'{opt.evaluation_folder}/pred.json'
        json.dump(ground_truth, open(gtFile, "w"))
        json.dump(prediction, open(dtFile, "w"))
        cocoGt = COCO(gtFile)
        cocoDt = cocoGt.loadRes(dtFile)
        prec = coco_eval(cocoGt, cocoDt)

        add_pr_curve(writer, "Val/PR", epoch * len(loader) + i, prec, cocoGt.getCatIds())

        return prec
    except Exception as e:
        print(e)
        return None
            

@torch.no_grad()
def val(meter, epoch, model, loader, loss_fn, writer, opt):
    device = next(model.parameters()).device
    model.eval()
    meter.reset()
    
    # initialize variables for metric calculation
    image_id = -1    
    gt_ann_id = -1
    ground_truth = empty_gt_dict()
    prediction = []

    for i, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        output = model(batch["image"])
        _, loss_dict = loss_fn(output, batch)

        # evolve variables for metric calculation
        image_id, gt_ann_id = build_annotations(output, batch, 
            image_id, gt_ann_id, ground_truth, prediction, opt)

        meter.update(loss_dict)
        meter.to_writer(writer, "Val", n_iter=epoch * len(loader) + i)

        if i % opt.val_vis_interval == 0:
            batch = {k: v[:1].detach().clone().cpu() for k, v in batch.items()}
            output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}
            
            add_dt(writer, "Val/DT", epoch * len(loader) + i, output, batch, opt)
            add_gt(writer, "Val/GT", epoch * len(loader) + i, output, batch, opt)
            add_hms(writer, "Val/HMS", epoch * len(loader) + i, output, batch)
    
    # save, load and evaluate
    gtFile = f'{opt.evaluation_folder}/gt.json'
    dtFile = f'{opt.evaluation_folder}/pred.json'
    json.dump(ground_truth, open(gtFile, "w"))
    json.dump(prediction, open(dtFile, "w"))
    cocoGt = COCO(gtFile)
    cocoDt = cocoGt.loadRes(dtFile)
    prec = coco_eval(cocoGt, cocoDt)

    add_pr_curve(writer, "Val/PR", epoch * len(loader) + i, prec, cocoGt.getCatIds())

    return prec

def stream_train(meter, count, model, optimizer, 
    loader, loss_fn, writer, opt):
    device = next(model.parameters()).device
    model.train()
    meter.reset()

    with tqdm(total=opt.val_interval) as pbar:
        for i, batch in enumerate(loader):
            if i == opt.val_interval:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(batch["image"])
            loss, loss_dict = loss_fn(output, batch)

            meter.update(loss_dict)
            meter.to_writer(writer, "Train", n_iter=count + i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % opt.train_vis_interval == 0:
                batch = {k: v[:1].detach().clone().cpu() for k, v in batch.items()}
                output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}
                
                add_dt(writer, "Train/DT", epoch * len(loader) + i, output, batch, opt)
                add_gt(writer, "Train/GT", epoch * len(loader) + i, output, batch, opt)
                add_hms(writer, "Train/HMS", epoch * len(loader) + i, output, batch)
            pbar.set_postfix(loss=loss.item())
            pbar.update()
        pbar.close()
    
    return count + opt.val_interval

def empty_gt_dict():
    return {
        "images": [], 
        "annotations": [], 
        "categories": [
            {"id": cid, "name": str(cid)} for cid in range(len(GROUPS)) 
        ]
    }

def build_annotations(output, batch, image_id, gt_ann_id, 
    ground_truth, prediction, opt):
    # b x k x 6; b x k x [bbox, score, cid]
    all_dets = decode(output, opt.k)  # unfiltered 

    gt_bboxes = batch["bboxes"].cpu().numpy()  # b x n_max x 4
    gt_cids = batch["cids"].cpu().numpy()  # b x n_max
    # to get valid batch entries we need a mask,
    # for each batch cpt_mask are all 1s and then all 0s
    cpt_mask = batch["cpt_mask"].cpu().numpy()  # b x n_max
    batch_size = cpt_mask.shape[0]

    for b in range(batch_size):  # for each batch(image)
        # 1 x k' x 6 with possibly different k' each iteration,
        # where k' is the number of detections(bboxes)
        dets = filter_dets(all_dets[b:b+1, ...], opt.model_score_threshold_low)
        dets[..., :4] = dets[..., :4] * opt.down_ratio  # opt.h x opt.w again
        dets = dets.cpu().squeeze(0).numpy()  # 1 x k' x 6 -> k' x 6
        
        image_id += 1
        # build prediction annotations
        for det in dets:  # each of shape: 6,
            prediction.append({
                "image_id": int(image_id),
                "category_id": int(det[5]),
                "bbox": list(map(_to_float, det[:4])),
                "score": _to_float(det[4]),
            })

        # build ground truth annotations
        ground_truth["images"].append({"id": int(image_id)})
        for idx in range(cpt_mask[b].sum()):
            gt_ann_id += 1
            area = gt_bboxes[b, idx, 2] * gt_bboxes[b, idx, 3]  # in pixels
            ground_truth["annotations"].append({
                "image_id": int(image_id),
                "category_id": int(gt_cids[b, idx]),
                "bbox": list(map(_to_float, gt_bboxes[b, idx, :])),
                "id": int(gt_ann_id),  # must be unique
                "iscrowd": 0,
                "area": int(area),
            })    

    return image_id, gt_ann_id

@torch.no_grad()
def stream_eval(meter, count, model, loader, loss_fn, writer, opt):
    device = next(model.parameters()).device
    model.eval()
    meter.reset()

    # initialize variables for metric calculation
    image_id = -1    
    gt_ann_id = -1
    ground_truth = empty_gt_dict()
    prediction = []

    for i, batch in enumerate(loader):
        if i == opt.val_len:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        output = model(batch["image"])
        _, loss_dict = loss_fn(output, batch)

        # evolve variables for metric calculation
        image_id, gt_ann_id = build_annotations(output, batch, 
            image_id, gt_ann_id, ground_truth, prediction, opt)

        meter.update(loss_dict)
        meter.to_writer(writer, "Val", n_iter=count + i)

        if i % opt.val_vis_interval == 0:
            batch = {k: v[:1].detach().clone().cpu() for k, v in batch.items()}
            output = {k: v[:1].detach().clone().cpu() for k, v in output.items()}
            
            add_dt(writer, "Val/DT", epoch * len(loader) + i, output, batch, opt)
            add_gt(writer, "Val/GT", epoch * len(loader) + i, output, batch, opt)
            add_hms(writer, "Val/HMS", epoch * len(loader) + i, output, batch)

    # save, load and evaluate
    gtFile = f'{opt.evaluation_folder}/gt.json'
    dtFile = f'{opt.evaluation_folder}/pred.json'
    json.dump(ground_truth, open(gtFile, "w"))
    json.dump(prediction, open(dtFile, "w"))
    cocoGt = COCO(gtFile)
    cocoDt = cocoGt.loadRes(dtFile)
    prec = coco_eval(cocoGt, cocoDt)

    add_pr_curve(writer, "Val/PR", epoch * len(loader) + i, prec, cocoGt.getCatIds())
    
    return count + opt.val_len, prec

def adjust_stream(loss, prec, cdistr, steps, count,
    remotes, writer, opt):
    # adjust number of scene objects and class distribution
    if loss < opt.loss_threshold:
        opt.num_objects += opt.objects_step
        opt.num_objects = max(opt.n_max, opt.num_objects)
        
        # with GROUPS we define our custom classes
        metrics = np.zeros((len(GROUPS), ))
        for cid, metric in enumerate(metrics):
            # precision over all IoUs 0.5-0.95 (single class)
            pr = ap_values(prec, klass=cid)  # Precision(Recalls)
            mAP = pr.mean()  # mean over recall values = mAP
            metric += mAP.item()
        min_metric_class = np.argmin(metrics)
        min_original_classes = GROUPS[min_metric_class]  # e.g. TLess: 1-30
        for c in min_original_classes:
            c -= 1  # we use zero based indices
            cdistr[c] += opt.distribution_step 
    elif steps > opt.max_train_step:
        opt.num_objects -= opt.objects_step
        opt.num_objects = min(1, opt.num_objects)
        # reset class distribution
        cdistr = {k:1 for k in range(30)}  # 1-30 -> 0-29
        steps = 0

    writer.add_scalar('Number of objects', opt.num_objects)
    add_cdistr(writer, tag, count, cdistr, opt)

    for remote in remotes:
        remote.send(num_objects=opt.num_objects, cdistr=cdistr)

    return steps
