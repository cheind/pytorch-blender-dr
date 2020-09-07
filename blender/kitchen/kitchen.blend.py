# Update python-path with current blend file directory,
# so that package `tless` can be found.
import bpy
import sys
import numpy as np
p = bpy.path.abspath("//")
if p not in sys.path:
    sys.path.append(p)

from blendtorch import btb
from kitchen import scene, annotation

def main():
    # Parse script arguments passed via blendtorch launcher
    btargs, remainder = btb.parse_blendtorch_args()

    cabinets = None

    def pre_anim():
        nonlocal cabinets
        cabinets = scene.create_scene(num_objects=20)
        
    def post_frame(off, pub, anim, cam):
        pub.publish(
            image=off.render(), 
            bboxes=annotation.bboxes(cam, cabinets),
            cids=[1]*len(cabinets)
        )
    
    def post_anim(anim):
        nonlocal cabinets
        scene.remove_objects()
        cabinets = None

    # Random seed worker
    np.random.seed(btargs.btseed)

    # Data source
    pub = btb.DataPublisher(btargs.btsockets['DATA'], btargs.btid)

    # Setup default image rendering
    cam = btb.Camera()
    off = btb.OffScreenRenderer(camera=cam, mode='rgb', gamma_coeff=2.2)
    #off.set_render_style(shading='RENDERED', overlays=False)
    # You might want to set a cache limit for blender, so that image
    # textures don't accumulate
    off.set_render_style(shading='RENDERED', overlays=False) 

    # Setup the animation and run endlessly
    anim = btb.AnimationController()
    anim.pre_animation.add(pre_anim)
    anim.post_frame.add(post_frame, off, pub, anim, cam)
    anim.post_animation.add(post_anim, anim)
    anim.play(use_animation=False)

main()