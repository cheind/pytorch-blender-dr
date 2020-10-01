from pathlib import Path
from torch.utils import data
from blendtorch import btt
import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_image(ax, img, cids, bboxes, visfracs, min_visfrac):
    ax.imshow(img, origin='upper')
    for cid, bbox, vfrac in zip(cids,bboxes,visfracs):
        if vfrac > min_visfrac:
            rect = patches.Rectangle(bbox[:2],bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            ax.text(bbox[0]-10, bbox[1]+10, f'C:{cid.item()}, V:{vfrac:.2f}', fontsize=8)

def update_elapsed(elapsed):
    fig, axs = plt.subplots()
    plt.plot(np.arange(len(elapsed)), elapsed)
    fig.savefig(f'./tmp/elapsed.png')
    plt.close(fig)

def iterate(dl, outpath, min_visfrac=0.2, save_interval=100):
    DPI=96
    tlast = time.time()
    times = []
    for step, item in enumerate(dl):
        if step % save_interval == 0:                      
            elapsed = (time.time()-tlast)
            print(f'Received batch #{step:05d}, took {elapsed:.2f} secs since last.')
            times.append(elapsed)
            tlast = time.time()
            update_elapsed(times)

            img, bboxes, cids, visfracs = item['image'], item['bboxes'], item['cids'], item['visfracs']
            H,W = img.shape[1], img.shape[2]
            fig = plt.figure(frameon=False, figsize=(W*2/DPI,H*2/DPI), dpi=DPI)
            axs = [fig.add_axes([0,0,0.5,0.5]), fig.add_axes([0.5,0.0,0.5,0.5]), fig.add_axes([0.0,0.5,0.5,0.5]), fig.add_axes([0.5,0.5,0.5,0.5])]
            for i in range(img.shape[0]):
                draw_image(axs[i], img[i], cids[i], bboxes[i], visfracs[i], min_visfrac=min_visfrac)
                axs[i].set_axis_off()
                axs[i].set_xlim(0,W-1)
                axs[i].set_ylim(H-1,0)
            fig.savefig(f'./{outpath}/output_{step}.png')
            plt.close(fig)  

def main():
    import argparse
    parser = argparse.ArgumentParser('Record blendtorch t-less datasets')
    parser.add_argument('--num-items', default=512, type=int)
    parser.add_argument('--num-instances', default=4, type=int)
    parser.add_argument('--json-config', help='JSON configuration file')
    parser.add_argument('--outpath', help='Output directory', default='tmp/')
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('scene', help='Scene to generate [tless|kitchen]')
    args = parser.parse_args()

    # Define how we want to launch Blender
    launch_args = dict(
        scene=Path(__file__).parent/'blender'/args.scene/f'{args.scene}.blend',
        script=Path(__file__).parent/'blender'/args.scene/f'{args.scene}.blend.py',
        num_instances=args.num_instances,
        named_sockets=['DATA'],
        blend_path="C:/Program Files/Blender Foundation/Blender 2.90",
    )

    if args.json_config:
        path = Path(args.json_config)
        assert path.exists()
        launch_args['instance_args'] = [['--json-config', args.json_config]] * args.num_instances

    Path(args.outpath).mkdir(exist_ok=True, parents=True)

    # Launch Blender
    with btt.BlenderLauncher(**launch_args) as bl:
        # Create remote dataset and limit max length to 16 elements.
        addr = bl.launch_info.addresses['DATA']
        ds = btt.RemoteIterableDataset(
            addr, max_items=args.num_items, #record_path_prefix=f'{args.outpath}/{args.scene}',
            timeoutms=30*1000)
        dl = data.DataLoader(ds, batch_size=4, num_workers=4)
        t = time.time()
        iterate(dl, args.outpath, min_visfrac=0.2, save_interval=args.save_interval)
        print(f'Finished in {time.time()-t} seconds.')

    if args.json_config:
        from shutil import copyfile
        copyfile(args.json_config, f'tmp/{args.scene}.json')

if __name__ == '__main__':
    # example call:
    # python record.py --num-items 64 tless --num-instances 4 --json-config record_config.json
    # set --save-interval 1 to plot each sample!
    main()
