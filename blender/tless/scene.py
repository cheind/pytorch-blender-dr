import bpy
import numpy as np
import supershape as sshape

SCN = bpy.context.scene
LAYER = bpy.context.view_layer

from .config import DEFAULT_CONFIG

def create_bsdf_material(basecolor=None):
    mat = bpy.data.materials.new("randommat")
    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    bsdf = nodes.get("Principled BSDF") 
    if basecolor is None:
        basecolor = np.random.uniform(0,1,size=3)
    bsdf.inputs["Base Color"].default_value = np.concatenate((basecolor, [1.]))
    return mat

def randomize_box_material():
    mat = bpy.data.materials['BoxMaterial']
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    mapping = nodes.get('Mapping')
    mapping.inputs['Rotation'].default_value = np.random.uniform(-1.0, 1.0, size=3)
    mapping.inputs['Scale'].default_value = np.random.uniform(0.1, 3.0, size=3)

def create_object(cfg=DEFAULT_CONFIG):
    # See https://blender.stackexchange.com/questions/135597/how-to-duplicate-an-object-in-2-8-via-the-python-api-without-using-bpy-ops-obje
    tcoll = SCN.collection.children['Objects']      
    gcoll = SCN.collection.children['Generated']
    
    templates = list(tcoll.objects)
    if cfg['scene.object_cls_prob'] is None:
        ids = np.arange(len(templates))
        p = np.ones(len(templates))
    else:
        d = cfg['scene.object_cls_prob']
        ids = list(map(int, d.keys()))
        p = np.array(list(d.values()))
    p /= p.sum()
    c = np.random.choice(ids, p=p)
    
    new_obj = templates[c].copy()
    new_obj.data = templates[c].data.copy()
    
    intensity = np.random.uniform(*cfg['scene.object_intensity_range'])
    new_obj.active_material = create_bsdf_material((intensity, intensity, intensity))
    gcoll.objects.link(new_obj)
    
    new_obj.location = np.random.uniform(low=cfg['scene.object_location_bbox'][0],high=cfg['scene.object_location_bbox'][1],size=(3))
    new_obj.rotation_euler = np.random.uniform(low=cfg['scene.object_rotation_range'][0], high=cfg['scene.object_rotation_range'][1],size=(3))
    return new_obj
    
def create_occluder(cfg=DEFAULT_CONFIG):
    coll = SCN.collection.children['Occluders']
    
    shape=(50,50)
    new_obj = sshape.make_bpy_mesh(shape, name='occ', coll=coll)
    new_obj.active_material = create_bsdf_material()
    
    params = np.random.uniform(
        low =[0.00,1,1,0.0,0.0, 0.0],
        high=[20.00,1,1,40,10.0,10.0],
        size=(2,6)
    )
    
    scale = np.random.uniform(0.1, 0.6, size=3)    
    x,y,z = sshape.supercoords(params, shape=shape)    
    sshape.update_bpy_mesh(x*scale[0], y*scale[1], z*scale[2], new_obj)
    
    new_obj.location = np.random.uniform(low=[-2, -2, 3],high=[2,2,6],size=(3))
    new_obj.rotation_euler = np.random.uniform(low=-np.pi, high=np.pi,size=(3))    
    SCN.rigidbody_world.collection.objects.link(new_obj)
    
    return new_obj

def remove_objects():
    coll = SCN.collection.children['Generated']
    for o in coll.objects:
        bpy.data.objects.remove(o, do_unlink=True)
    coll = SCN.collection.children['Occluders']
    for o in coll.objects:
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.outliner.orphans_purge()

def apply_physics_to(objs, enabled=False, collision_shape='BOX', friction=0.5, linear_damp=0.05, angular_damp=0.1):
    for obj in objs:
        obj.rigid_body.enabled = enabled
        obj.rigid_body.collision_shape = collision_shape
        obj.rigid_body.friction = friction
        obj.rigid_body.linear_damping = linear_damp
        obj.rigid_body.angular_damping = angular_damp

def create_scene(cfg=DEFAULT_CONFIG):    
    N = cfg['scene.num_entities']
    b = np.random.binomial(N, 1-cfg['scene.prob_occluder'],size=1)
    M = N - b.sum()

    objs = [create_object(cfg) for _ in range(b.sum())]
    occs = [create_occluder(cfg) for _ in range(N - b.sum())]
    
    apply_physics_to(
        objs,
        enabled=True,
        linear_damp=cfg['physics.linear_damp'],
        friction=cfg['physics.friction'],
        angular_damp=cfg['physics.angular_damp'])
        
    apply_physics_to(
        occs,
        collision_shape='CONVEX_HULL',
        enabled=True,
        linear_damp=cfg['physics.linear_damp'],
        friction=cfg['physics.friction'],
        angular_damp=cfg['physics.angular_damp'])
        
    randomize_box_material()
        
    return objs, occs