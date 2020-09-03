# Update python-path with current blend file directory,
# so that package `tless` can be found.
import bpy
import sys
p = bpy.path.abspath("//")
if p not in sys.path:
    sys.path.append(p)

import numpy as np
from blendtorch import btb
from tless import scene
import re

def main():
    # Parse script arguments passed via blendtorch launcher
    btargs, remainder = btb.parse_blendtorch_args()

    objs, occs = None, None
    reg = r'^obj\_(\d+)\.'

    def bboxes(cam, objs):
        bbox_xy = cam.bbox_object_to_pixel(*objs).reshape(-1,8,2)
        minc = bbox_xy.min(1)
        maxc = bbox_xy.max(1)
        wh = maxc-minc
        return np.concatenate((minc,wh), -1)

    def classids(objs):
        return np.array(
            [int(re.match(reg, obj.name).group(1)) for obj in objs],
            dtype=np.long)
    
    def pre_anim():
        nonlocal objs, occs
        objs, occs = scene.create_scene()
        
    def post_frame(off, pub, anim, cam):
        if anim.frameid == 2:        
            pub.publish(
                image=off.render(), 
                bboxes=bboxes(cam, objs),
                cids=classids(objs)
            )

    def post_anim(anim):
        nonlocal objs, occs
        scene.remove_objects()
        objs, occs = None, None
        
    # Make sure every Blender has its own random seed
    np.random.seed(btargs.btseed)

    # Speed up physics
    bpy.context.scene.rigidbody_world.time_scale = 100

    # Data source
    pub = btb.DataPublisher(btargs.btsockets['DATA'], btargs.btid)

    # Setup default image rendering
    cam = btb.Camera()
    off = btb.OffScreenRenderer(camera=cam, mode='rgb', gamma_coeff=2.2)
    off.set_render_style(shading='RENDERED', overlays=False)

    # Setup the animation and run endlessly
    anim = btb.AnimationController()
    anim.pre_animation.add(pre_anim)
    anim.post_frame.add(post_frame, off, pub, anim, cam)
    anim.post_animation.add(post_anim, anim)
    # Cant use animation system here
    anim.play(frame_range=(1,3), num_episodes=-1, use_animation=False)

    

main()