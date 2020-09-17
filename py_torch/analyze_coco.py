from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import copy


def main():
    cocoGt = COCO('evaluation/gt.json')
    cocoDt = cocoGt.loadRes('evaluation/pred.json')

    catids = cocoGt.getCatIds()
    # print(cocoGt.loadImgs([0]))

    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.evaluate()
    params = copy.deepcopy(cocoEval.params)
    
    print()
    print('TOTAL')
    print('----------------------------')    
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate(p=params)
    cocoEval.summarize()

    for c in catids:
        print()
        print(f'CLASS {c}')
        print('----------------------------')   
        cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
        cocoEval.params.catIds = [c]
        cocoEval.evaluate() 
        cocoEval.accumulate()
        cocoEval.summarize()
        
    
    
if __name__ == '__main__':
    main()