from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse


def main():
    cocoGt = COCO('evaluation/gt.json')
    cocoDt = cocoGt.loadRes('evaluation/pred.json')

    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
if __name__ == '__main__':
    main()