from pathlib import Path
from torch.utils import data
from blendtorch import btt

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def iterate(dl):
    DPI=96
    for step, item in enumerate(dl):
        img, bboxes, cids = item['image'], item['bboxes'], item['cids']
        H,W = img.shape[1], img.shape[2]
        fig = plt.figure(frameon=False, figsize=(W*2/DPI,H*2/DPI), dpi=DPI)
        axs = [fig.add_axes([0,0,0.5,0.5]), fig.add_axes([0.5,0.0,0.5,0.5]), fig.add_axes([0.0,0.5,0.5,0.5]), fig.add_axes([0.5,0.5,0.5,0.5])]
        for i in range(img.shape[0]):
            axs[i].imshow(img[i], origin='upper')
            for cid, bbox in zip(cids[i],bboxes[i]):
                rect = patches.Rectangle(bbox[:2],bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')
                axs[i].add_patch(rect)
                axs[i].text(bbox[0]+10, bbox[1]+10, f'Class {cid.item()}', fontsize=18)
            axs[i].set_axis_off()
            axs[i].set_xlim(0,W-1)
            axs[i].set_ylim(H-1,0)
        fig.savefig(f'./tmp/output_{step}.png')
        plt.close(fig)

def main():
    import argparse
    parser = argparse.ArgumentParser('Record blendtorch t-less datasets')
    parser.add_argument('--num-items', default=512, type=int)
    parser.add_argument('--num-instances', default=4, type=int)
    parser.add_argument('--prefix-name', default='tless')    
    parser.add_argument('--json-config', help='JSON configuration file')
    args = parser.parse_args()

    # Define how we want to launch Blender
    launch_args = dict(
        scene=Path(__file__).parent/'blender'/'tless.blend',
        script=Path(__file__).parent/'blender'/'tless.blend.py',
        num_instances=args.num_instances, 
        named_sockets=['DATA'],
    )

    if args.json_config:
        path = Path(args.json_config)
        assert path.exists()
        launch_args['instance_args'] = [['--json-config', args.json_config]] * args.num_instances


    # Launch Blender
    with btt.BlenderLauncher(**launch_args) as bl:
        # Create remote dataset and limit max length to 16 elements.
        addr = bl.launch_info.addresses['DATA']
        ds = btt.RemoteIterableDataset(addr, max_items=args.num_items, record_path_prefix=f'tmp/{args.prefix_name}')
        dl = data.DataLoader(ds, batch_size=4, num_workers=4) # bug when num_workers = 4, the batch size is only one then??
        iterate(dl)

    if args.json_config:
        from shutil import copyfile
        copyfile(args.json_config, f'tmp/{args.prefix_name}.json')

if __name__ == '__main__':
    main()