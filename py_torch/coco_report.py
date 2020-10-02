import argparse
from glob import glob
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import contextlib
import itertools
import io


import seaborn as sns
sns.set_context('paper', rc={'lines.linewidth': 1.5}, font_scale=1.1)

class COCOQuery:
    def __init__(self, cocoGt, cocoDt):        
        with contextlib.redirect_stdout(None):
            self.cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
            self.cocoEval.evaluate()
            self.cocoEval.accumulate()
            self.cocoEval.summarize()
        self.cocoGt = cocoGt
        self.precision = self.cocoEval.eval['precision']
        self.params = self.cocoEval.params
        # Only interested in maxdets == 100 and all areas
        self.aind = [i for i, aRng in enumerate(self.params.areaRngLbl) if aRng == 'all'][0]
        self.mind = [i for i, mDet in enumerate(self.params.maxDets) if mDet == 100][0]
    
    def __str__(self):
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            self.cocoEval.summarize()
            return buf.getvalue()

    @property
    def recall_values(self):
        return self.params.recThrs

    @property
    def class_ids(self):
        return self.cocoGt.getCatIds()

    def ap(self, iou=None, klass=None):
        # TxRxKxAxM
        prec = self.precision[...,self.aind, self.mind]
        # TxRxK
        if iou is None:
            prec = prec.mean(0)
        else:
            t = np.where(iou == self.params.iouThrs)[0]
            prec = prec[t]
        # RxK
        if klass is None:
            prec = prec.mean(-1)
        else:
            prec = prec[...,klass]

        return prec


class MultipleRunCOCOQuery:
    def __init__(self):
        self.runs = []

    def append(self, query):
        self.runs.append(query)

    @property
    def recall_values(self):
        return self.runs[0].recall_values

    @property
    def class_ids(self):
        return self.runs[0].class_ids

    def ap(self, iou=None, klass=None, return_std=False):
        prec = np.stack([q.ap(iou=iou, klass=klass) for q in self.runs], 0)
        if return_std:
            return prec.mean(0), prec.std(0)
        else:
            return prec.mean(0)

def draw_roc(cocoQuery, ax=None, show_classes=True, show_error=False):
    if ax is None:
        ax = plt.gca()

    marker = itertools.cycle(('.', '<', 's', 'p', 'D', '*'))         
    x = cocoQuery.recall_values
    for c in cocoQuery.class_ids:
        y = cocoQuery.ap(klass=c)
        ax.plot(x, y, next(marker), ls='-', ms=6, label=f'mAP[class={c}]={y.mean():.3f}', markevery=20) #markerfacecolor='none'
    ymean, ystd = cocoQuery.ap(return_std=True)
    ax.errorbar(x, ymean, yerr=ystd, errorevery=10, ls='--', color='k', label=f'mAP={ymean.mean():.3f}', capsize=3)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    return ax

def query_summary(query):
    d = {
        'AP[0.5:0.95|cls=all]' : query.ap().mean(),
        'AP[     0.5|cls=all]' : query.ap(iou=0.5).mean(),
        'AP[    0.75|cls=all]' : query.ap(iou=0.75).mean(),
    }
    for c in query.class_ids:
        d[f'AP[0.5:0.95|cls={c}]'] = query.ap(klass=c).mean()
    return d
    
def glob_files(path):
    gt = path / 'gt.json'
    assert gt.exists()
 
    runs = glob(str(path / 'pred_*.json'))
    if len(runs) == 0:
        pred = path / 'pred.json'
        if pred.exists():
            runs = [str(pred)]
    
    return gt, runs
    
def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('path')
    args = parser.parse_args()

    print(f'Analyzing {args.path}')
    p = Path(args.path)
    assert p.exists() and p.is_dir()

    gt, runs = glob_files(p)
    mquery = MultipleRunCOCOQuery()

    per_run_summary = []

    with contextlib.redirect_stdout(None):
        cocoGt = COCO(str(gt))    
        for idx, r in enumerate(runs):
            cocoDt = cocoGt.loadRes(str(r))
            query = COCOQuery(cocoGt, cocoDt)
            per_run_summary.append(query_summary(query))
            mquery.append(query)

    

    df = pd.DataFrame(per_run_summary)
    df.index.name = 'Run'
    df.to_csv(str(p/'coco_analyze_runs.csv'))
    print(df['AP[0.5:0.95|cls=all]'].describe())

    summary = query_summary(mquery)
    df = pd.DataFrame([summary])
    df.to_csv(str(p/'coco_analyze_summary.csv'))
    print(df)

    fig, ax = plt.subplots()
    draw_roc(mquery, ax)
    sns.despine()
    plt.legend(loc='upper right')
    plt.savefig(str(p/'coco_analyze.svg'))

if __name__ == '__main__':
    main()