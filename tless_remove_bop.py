'''Filters BoP challenge data from real test data.

Done, so we can avoid training on BoP data when training on real images.
'''

import argparse
import json
import shutil
from pathlib import Path

def process_scene(super_scene, subset_scene):
    super_gt_path = super_scene / 'scene_gt.json'
    subset_gt_path = subset_scene / 'scene_gt.json'
    with open(super_gt_path, 'r') as fp:
        super_gt = json.loads(fp.read())
    with open(subset_gt_path, 'r') as fp:
        subset_gt = json.loads(fp.read())

    isect = set(super_gt.keys()).intersection(set(subset_gt.keys()))
    print(f'--Removing {len(isect)} keys')
    
    for k in isect:
        del super_gt[k]

    bak = super_gt_path.with_suffix('.orig.json')
    if not bak.exists():
        print(f'--Creating backup {bak.name}')
        shutil.copy(super_gt_path, bak)

    with open(super_gt_path, 'w') as fp:
        fp.write(json.dumps(super_gt))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('subset', help='Path to BoP dataset')
    parser.add_argument('superset', help='Path to superset of real test data')
    args = parser.parse_args()

    subset_path = Path(args.subset)
    superset_path = Path(args.superset)

    assert subset_path.exists(), 'BoP path does not exist'
    assert superset_path.exists(), 'Superset of real data does not exist'

    superset_scenes = [f for f in superset_path.iterdir() if f.is_dir()]
    for superset_s in superset_scenes:
        subset_s = subset_path/superset_s.name 
        if subset_s.exists():
            print(f'Processing {subset_s.name}')
            process_scene(superset_s, subset_s)
        else:
            print(f'Skipping {subset_s.name}')



if __name__ == '__main__':
    main()