import bpy
import numpy as np

# Update python-path with current blend file directory,
# so that package `tless` can be found.
import sys
p = bpy.path.abspath("//")
if p not in sys.path:
    sys.path.append(p)

from blendtorch import btb
from tless import scene, annotation
from tless.config import DEFAULT_CONFIG

def main(cfg):
    # Parse script arguments passed via blendtorch launcher
    btargs, remainder = btb.parse_blendtorch_args()

    objs, occs = None, None

    def pre_anim():
        nonlocal objs, occs
        objs, occs = scene.create_scene(cfg)
        
    def post_frame(off, pub, anim, cam):
        if anim.frameid == 2: 
            # Instead of generating just one image per simulation,
            # we generate N images from the same scene using 
            # random camera poses.       
            for _ in range(cfg['camera.num_images']):
                pub.publish(
                    image=off.render(), 
                    bboxes=annotation.bboxes(cam, objs),
                    cids=annotation.classids(objs)
                )
                lfrom = btb.utils.random_spherical_loc(
                    radius_range=cfg['camera.radius_range'], 
                    theta_range=cfg['camera.theta_range']
                )
                cam.look_at(look_at=cfg['camera.lookat'], look_from=lfrom)

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
    

main(DEFAULT_CONFIG)