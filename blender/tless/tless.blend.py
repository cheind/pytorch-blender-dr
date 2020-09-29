import bpy
import numpy as np
import argparse
import json

# Update python-path with current blend file directory,
# so that package `tless` can be found.
import sys
p = bpy.path.abspath("//")
if p not in sys.path:
    sys.path.append(p)

from blendtorch import btb
from tless import scene, annotation
from tless.config import DEFAULT_CONFIG

def parse_additional_args(remainder):
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-config')
    return parser.parse_args(remainder)

def main():
    # Parse script arguments passed via blendtorch launcher
    btargs, remainder = btb.parse_blendtorch_args()
    otherargs = parse_additional_args(remainder)

    if otherargs.json_config is not None:
        print('Got custom configuration file.')
        with open(otherargs.json_config, 'r') as fp:
            cfg = json.loads(fp.read())
    else:
        cfg = DEFAULT_CONFIG        

    step = 0
    objs, occs = None, None
    
    def pre_anim():
        nonlocal objs, occs        
        objs, occs = scene.create_scene(pre_gen_data, cfg)
        
    def post_frame(off, pub, anim, cam, pre_gen_data):
        if anim.frameid == 2: 
            # Instead of generating just one image per simulation,
            # we generate N images from the same scene using 
            # random camera poses.       
            for _ in range(cfg['camera.num_images']):
                bboxes = annotation.bboxes(cam, objs, pre_gen_data.simplified_xyz)
                visfracs = annotation.compute_visfracs(cam, objs, bboxes)
                pub.publish(
                    image=off.render(), 
                    bboxes=bboxes,
                    visfracs=visfracs,
                    cids=annotation.classids(objs)
                )
                lfrom = btb.utils.random_spherical_loc(
                    radius_range=cfg['camera.radius_range'], 
                    theta_range=cfg['camera.theta_range']
                )
                cam.look_at(look_at=cfg['camera.lookat'], look_from=lfrom)

    def post_anim(anim):
        nonlocal objs, occs, step
        scene.remove_objects(objs, occs)
        objs, occs = None, None

    # pre-generated reusable data
    pre_gen_data = scene.PreGeneratedData(
        max_occluders=cfg['scene.num_objects'],
        max_materials=cfg['scene.num_objects'],
        num_faces_simplified=300,
        occ_shape=(50,50)
    )
        
    # Make sure every Blender has its own random seed
    np.random.seed(btargs.btseed)

    # Speed up physics
    bpy.context.scene.rigidbody_world.time_scale = 100

    # Data source
    pub = btb.DataPublisher(btargs.btsockets['DATA'], btargs.btid)

    # Setup default image rendering
    cam = btb.Camera()
    #off = btb.OffScreenRenderer(camera=cam, mode='rgb', gamma_coeff=2.2)
    off = btb.Renderer(btargs.btid, camera=cam, mode='rgb', gamma_coeff=2.2)
    #off.set_render_style(shading='RENDERED', overlays=False)

    # Setup the animation and run endlessly
    anim = btb.AnimationController()
    anim.pre_animation.add(pre_anim)
    anim.post_frame.add(post_frame, off, pub, anim, cam, pre_gen_data)
    anim.post_animation.add(post_anim, anim)
    # Cant use animation system here
    anim.play(frame_range=(1,3), num_episodes=-1, use_animation=False)
    

main()