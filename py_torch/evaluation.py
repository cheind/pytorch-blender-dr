from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import logging
import time
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast

from .constants import CATEGORIES
from .utils import FileStream
from .decode import decode, filter_dets
from .visu import render
from .utils import _to_float


def create_gt_anns(rbg_relpaths, all_bboxes, all_category_ids, 
    path):
    """Create ground truth annotations for evaluation. 

    ATTENTION: BBOXES IN COCO FORMAT

    Args:
        rbg_relpaths (list): relative paths to image files
        all_bboxes (list): bboxes for each image
        all_category_ids (list): class ids for bboxes  
        path (str): path of annotations file. 
    """
    gt = {"categories": CATEGORIES}
    images = []
    annotations = []
    id = 0
    logging.info(f"Save gt. ann. file @ {path}")
    start_time = time.time()

    for image_id, (fpath, bboxes, category_ids) in enumerate(zip(rbg_relpaths, all_bboxes, all_category_ids)):
        images.append({"id": int(image_id), "file_name": str(fpath)})
 
        # iterate over all bboxes for the current image
        for category_id, bbox in zip(category_ids, bboxes):
            # bbox format: x, y, w, h
            area = bbox[2] * bbox[3]  # in pixels

            annotations.append({
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": list(map(_to_float, bbox)),
                 # each annotation must have a unique id
                "id": int(id),  
                "iscrowd": 0,
                "area": int(area),
            })
            id += 1

    gt.update({"images": images, "annotations": annotations})
    json.dump(gt, open(path, "w"))
    logging.info(f"Elapsed time: {time.time() - start_time} s")

def evaluate(gtFile: str, dtFile: str, annType = 'bbox'):
    cocoGt = COCO(gtFile)  # .json files
    cocoDt = cocoGt.loadRes(dtFile)  # .json files

    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

@torch.no_grad()
def evaluate_model(model, dl, opt):
    model.eval()
    device = next(model.parameters()).device
    pred = []

    for (i, batch) in tqdm(enumerate(dl), desc="Evaluation", total=len(dl)):
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast(enabled=opt.amp):
            out = model(batch["image"])  # 1 x 3 x h x w
        dets = decode(out, opt.k)  # 1 x k x 6
        dets = filter_dets(dets, opt.model_score_threshold_low)  # 1 x k' x 6
        
        #import pdb; pdb.set_trace()

        image_gt = batch["image_gt"]  # 1 x h x w x 3, original image
        dets[..., :4] = dets[..., :4] * opt.down_ratio  # 512 x 512 space dets

        shape_pp = batch["shape_pp"]  # 1 x 2
        h_gt, w_gt = image_gt.size(1), image_gt.size(2)

        # Pre padded image height and width
        h_pp, w_pp = shape_pp[0]  
        x_scale = w_gt / w_pp
        y_scale = h_gt / h_pp

        # Scale bboxes to match original image space
        dets[..., 0] *= x_scale  # x
        dets[..., 1] *= y_scale  # y
        dets[..., 2] *= x_scale  # w
        dets[..., 3] *= y_scale  # h

        if opt.render_detections:
            render(image_gt, dets, opt, show=False, save=True, 
                path=f"{opt.detection_folder}/{i:05d}.png", 
                denormalize=False)

        # Create json results for AP evaluation
        image_id = int(batch["image_id"])
        dets = dets.squeeze(0).cpu().numpy()  # k' x 6

        bboxes = dets[..., :4]  # k' x 4
        scores = dets[..., 4]  # k',
        cids = dets[..., 5]  # k',

        for bbox, cid, score in zip(bboxes, cids, scores):
            pred.append({
                "image_id": image_id,
                "category_id": int(cid),
                "bbox": list(map(_to_float, bbox)),
                "score": _to_float(score),
            })
        
    # Save json results for evaluation
    json.dump(pred, open(f"{opt.evaluation_folder}/pred.json", "w"))
    # Save console output
    with FileStream(f"{opt.evaluation_folder}/mAP.txt"):
        evaluate(f"{opt.evaluation_folder}/gt.json", 
            f"{opt.evaluation_folder}/pred.json")

if __name__ == "__main__":
    gt = {
        "images":[
            {"id": 73}, 
            {"id": 72}
        ],
        "annotations":[
            {
                "image_id":73,
                "category_id":1,
                "bbox":[10,10,50,100],
                "id":56786,
                "iscrowd": 0,
                "area": 100
            },
            {
                "image_id":73,
                "category_id":2,
                "bbox":[10,20,60,100],
                "id":45645,
                "iscrowd": 0,
                "area": 5000.3
            },
            {
                "image_id":72,
                "category_id":3,
                "bbox":[10,20,60,100],
                "id":345234,
                "iscrowd": 0,
                "area": 10000
            },
        ],
        "categories": [
            {"id": 1, "name": "person"}, 
            {"id": 2, "name": "bicycle"}, 
            {"id": 3, "name": "car"}
        ]
    }

    dets = [
        {
            "image_id":73,
            "category_id":1,
            "bbox":[10,10,50,100],
            "score":0.8
        },
        {
            "image_id":73,
            "category_id":2,
            "bbox":[10,20,60,100],
            "score":0.9
        },
        {
            "image_id":72,
            "category_id":1,
            "bbox":[10,20,60,100],
            "score":0.7
        },
    ]

    gtFile = './ann.json'
    dtFile='./res.json'
    json.dump(gt, open(gtFile, "w"))
    json.dump(dets, open(dtFile, "w"))
    evaluate(gtFile, dtFile)

    """
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.667
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.667
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.667
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 1.000
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 1.000
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.667
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.667
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.667
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 1.000
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 1.000
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
    """
