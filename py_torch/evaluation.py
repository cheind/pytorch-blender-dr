from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import logging
import time

from .main import CATEGORIES


def create_gt_anns(rbg_relpaths, all_bboxes, all_category_ids, 
    path):
    """Create ground truth annotations for evaluation. 

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

    logging.info(f"Save ground truth annotations at: {path}")
    logging.info("Assume COCO bbox format: xmin, ymin, width, height")

    start_time = time.time()

    for image_id, (fpath, bboxes, category_ids) in enumerate(zip(rbg_relpaths, all_bboxes, all_category_ids)):
        images.append({"id": int(image_id), "file_name": str(fpath)})

        is_crowd = 0  # crowd annotation? NO! => 0

        # iterate over all bboxes for the current image
        for category_id, bbox in zip(category_ids, bboxes):
            # bbox format: x, y, w, h
            # area doesn't have to be precise, just to get metrics
            # over different object sizes: small, medium, large
            area = bbox[2] * bbox[3]  # number of pixels

            annotations.append({
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": bbox.tolist(),
                "id": int(id),  # each annotation has to have a unique id
                "iscrowd": int(is_crowd),
                "area": int(area),
            })
            id += 1

    gt.update({"images": images, "annotations": annotations})
    json.dump(gt, open(path, "w"))

    logging.info(f"Saved annotations. Elapsed time: {time.time() - start_time} s")


def evaluate(gtFile, dtFile, annType = 'bbox'):
    """ Evaluates content of ground truth and detection .json files. """
    cocoGt = COCO(gtFile)
    cocoDt = cocoGt.loadRes(dtFile)

    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == "__main__":
    # ground truth example
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

    # detection example
    res = [
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

    # save ground truths
    json.dump(gt, open(gtFile, "w"))

    # save detections
    json.dump(res, open(dtFile, "w"))

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
