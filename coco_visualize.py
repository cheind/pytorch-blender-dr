import argparse
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
import numpy as np
from tqdm import tqdm

import seaborn as sns
sns.set_context('paper', rc={'lines.linewidth': 1.5}, font_scale=1.1)
sns.set_style('whitegrid', {'font.family':'serif', 'font.serif':'Times New Roman'})

LINEWIDTH=1.5
FONTSIZE=10
DPI=96

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def draw_bbox(ax, bbox, text=None, linestyle=None, color=None):
    [bbox_x, bbox_y, bbox_w, bbox_h] = bbox
    poly = [
        [bbox_x, bbox_y], 
        [bbox_x, bbox_y+bbox_h],
        [bbox_x+bbox_w, bbox_y+bbox_h], 
        [bbox_x+bbox_w, bbox_y]
    ]    
    np_poly = np.array(poly).reshape((4,2))
    # rect = Polygon(np_poly, linestyle=linestyle, facecolor='none', edgecolor=color, linewidth=LINEWIDTH)
    # rect.set_path_effects([
    #         path_effects.Stroke(linewidth=2, foreground="white"),
    #         path_effects.Normal()
    #     ])
    # ax.add_patch(rect)
    ax.add_patch(Polygon(np_poly, linestyle=linestyle, facecolor='none', edgecolor=color, linewidth=LINEWIDTH))

    if text is not None:
        t = ax.text(bbox_x, bbox_y-2, text, fontsize=FONTSIZE, color='white')        
        t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

def render(img_id, coco_gt, coco_dt, ds_path, cat_ids=None, legend=True, min_score=0.1, scale=1.0, show_score=False):
    if cat_ids is None:
        cat_ids = coco_gt.getCatIds()
        
    cmap = get_cmap(len(cat_ids))
    colors = ['darkred','darkorange','darkgreen','darkblue','blueviolet','magenta']
    
    # make colors appear lighter to add more contrast to dark areas
    import matplotlib
    colors = [np.array(matplotlib.colors.to_rgb(c)) for c in colors]
    tint = 0.15
    colors = [np.clip(c + tint, a_min=0, a_max=1) for c in colors]

    img_meta = coco_gt.loadImgs([img_id])[0]
    img = plt.imread(str(ds_path/img_meta['file_name']))
    
    gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None))
    dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None))
    dt_anns = [a for a in dt_anns if a['score']>=min_score]
    
    fig = plt.figure(figsize=(img.shape[1]/DPI*scale, img.shape[0]/DPI*scale), dpi=DPI, frameon=False)
    ax = fig.add_axes([0,0,1.,1.])    
    ax.imshow(img, origin='upper')
    ax.set_axis_off()
    ax.autoscale(False)
    
    for ann in gt_anns:
        draw_bbox(ax, ann['bbox'], text=None, linestyle='-', color=colors[ann['category_id']])
        
    for ann in dt_anns:  # linestyle='--'
        text = f'{ann["score"]:.2f}' if show_score else None
        draw_bbox(ax, ann['bbox'], text=text, linestyle='--', color=colors[ann['category_id']])
        
    if legend:
        legend_elements = [Line2D([0], [0], linestyle='-', color='k', label='Groundtruth'),
                           Line2D([0], [0], linestyle='--', color='k', label='Detection')]    
        legend_elements.extend([Line2D([0], [0], marker='o', color=colors[c], label=f'Cat.{c}') for c in cat_ids])
        ax.legend(handles=legend_elements, loc='upper center', ncol=4, prop={'size': FONTSIZE})
        
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dspath')
    parser.add_argument('gtpath')
    parser.add_argument('predpath')
    parser.add_argument('--idx', type=int, default=None)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--min-score', type=float, default=0.3)
    args = parser.parse_args()

    # python coco_visualize.py /mnt/data/tless_test_real/test_primesense btorch_08_gt.json btorch_08_pred.json --save

    dspath = Path(args.dspath)
    gtpath = Path(args.gtpath)
    predpath = Path(args.predpath)

    assert dspath.exists() and dspath.is_dir()
    assert gtpath.exists() and gtpath.is_file()
    assert predpath.exists() and predpath.is_file()

    cocoGt = COCO(str(gtpath))
    cocoDt = cocoGt.loadRes(str(predpath))

    if args.idx != None:
        ids = [args.idx]
    else:
        ids = cocoGt.getImgIds()

    with tqdm(total=len(ids)) as pbar:
        for idx in ids:
            fig = render(idx, cocoGt, cocoDt, dspath, min_score=args.min_score, show_score=True, scale=args.scale, legend=False)
            if args.save:
                fig.savefig(f'tmp/cocoeval_{idx:04d}.png')
            else:
                plt.show()
            plt.close(fig)
            pbar.update()
        pbar.refresh()

        
if __name__ == '__main__':
    main()