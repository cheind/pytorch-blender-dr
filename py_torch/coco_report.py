import argparse
from glob import glob
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import contextlib
import itertools
from collections.abc import Iterable
import io


import seaborn as sns
sns.set_context('paper', rc={'lines.linewidth': 1.5}, font_scale=1.1)
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})

def coco_eval(cocoGts, cocoDts):
    def _run(cocoGt, cocoDt):
        with contextlib.redirect_stdout(None):
            cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # Only interested in maxdets == 100 and all areas
            aind = [i for i, aRng in enumerate(cocoEval.params.areaRngLbl) if aRng == 'all'][0]
            mind = [i for i, mDet in enumerate(cocoEval.params.maxDets) if mDet == 100][0]
        return cocoEval.eval['precision'][...,aind,mind] # TxRxK tensor
    
    if not isinstance(cocoDts, Iterable):
        cocoDts = [cocoDts]
    if not isinstance(cocoGts, Iterable):
        cocoGts = [cocoGts]*len(cocoDts)
    assert len(cocoGts) == len(cocoDts)

    precs = np.stack([_run(gt, dt) for gt,dt in zip(cocoGts, cocoDts)], 0) # YxTxRxK tensor
    return precs

def ap_values(prec, run=None, iou=None, klass=None, return_std=False):
    params = Params(iouType='bbox')
    # YxTxRxK
    if iou is None:
        prec = prec.mean(1) # average out iou (T)
    else:
        t = np.where(iou == params.iouThrs)[0]
        prec = prec[:,t] # select iou
    # YxRxK
    if klass is None:
        prec = prec.mean(-1) # average out klass (K)
    else:
        prec = prec[...,klass] # select klass
    if run is None:
        # YxR
        if return_std:
            return prec.mean(0), prec.std(0)
        else:
            return prec.mean(0)
    else:
        assert not return_std
        return prec[run]

def draw_roc(prec, class_ids, ax=None, show_classes=True, show_error=False):
    if ax is None:
        ax = plt.gca()

    params = Params(iouType='bbox')
    marker = itertools.cycle(('.', '<', 's', 'p', 'D', '*'))         
    x = params.recThrs
    for c in class_ids:
        y = ap_values(prec, klass=c)
        ax.plot(x, y, next(marker), ls='-', ms=6, label=f'mAP[class={c}]={y.mean():.3f}', markevery=20) #markerfacecolor='none'
    ymean, ystd = ap_values(prec, return_std=True)
    ax.errorbar(x, ymean, yerr=ystd, errorevery=10, ls='--', color='k', label=f'mAP={ymean.mean():.3f}', capsize=3)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    return ax

def ap_summary(prec, class_ids, run=0):
    d = {
        'AP[0.5:0.95|cls=all]' : ap_values(prec, run=run).mean(),
        'AP[     0.5|cls=all]' : ap_values(prec, run=run, iou=0.5).mean(),
        'AP[    0.75|cls=all]' : ap_values(prec, run=run, iou=0.75).mean(),
    }
    for c in class_ids:
        d[f'AP[0.5:0.95|cls={c}]'] = ap_values(prec, run=run, klass=c).mean()
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
    parser.add_argument('--load-precisions', action='store_true')
    args = parser.parse_args()

    print(f'Analyzing {args.path}')
    p = Path(args.path)
    assert p.exists() and p.is_dir()

    gt, runs = glob_files(p)
    per_run_summary = []

    with contextlib.redirect_stdout(None):
        cocoGt = COCO(str(gt))
        cocoDts = [cocoGt.loadRes(str(r)) for r in runs]
        if not args.load_precisions:
            prec = coco_eval(cocoGt, cocoDts) # YxTxRxK tensor
            np.save(str(p/'coco_run_precisions.npy'), prec)
        else:
            prec = np.load(str(p/'coco_run_precisions.npy'))
        
    run_summaries = [ap_summary(prec, cocoGt.getCatIds(), run=r) for r in range(len(runs))]
    
    df = pd.DataFrame(run_summaries)
    df.index.name = 'Run'
    df.to_csv(str(p/'coco_analyze_runs.csv'))
    print(df['AP[0.5:0.95|cls=all]'].describe())

    summary = ap_summary(prec, cocoGt.getCatIds(), run=None)
    df = pd.DataFrame([summary])
    df.to_csv(str(p/'coco_analyze_summary.csv'))
    print(df)

    fig, ax = plt.subplots()
    draw_roc(prec, cocoGt.getCatIds(), ax)
    sns.despine()
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.savefig(str(p/'coco_analyze.svg'))
    plt.savefig(str(p/'coco_analyze.pdf'))

if __name__ == '__main__':
    main()