import bpy
from mathutils import Euler, Vector
import numpy as np

SCN = bpy.context.scene
LAYER = bpy.context.view_layer

def create_object(parent):
    tcoll = SCN.collection.children['Templates']      
    gcoll = SCN.collection.children['Generated']      
    templates = list(tcoll.objects)
    c = np.random.choice(len(templates))
    
    new_obj = templates[c].copy()
    new_obj.data = templates[c].data.copy()
    new_obj.active_material = templates[c].active_material.copy()
        
    #tcoll.objects.unlink(new_obj)
    gcoll.objects.link(new_obj)
    
    bpy.ops.object.select_all(action='DESELECT')
    parent.select_set(True)
    new_obj.select_set(True)
    bpy.context.view_layer.objects.active = parent
    bpy.ops.object.parent_set(type='OBJECT', xmirror=False, keep_transform=False)
    
    return new_obj

def compute_world_bbox(obj):
    bbox_corners = np.stack([obj.matrix_world @ Vector(corner) for corner in obj.bound_box], 0)
    return np.min(bbox_corners, 0), np.max(bbox_corners, 0)

def change_object_properties(obj, conveyor_pos, **props):

    off = np.random.uniform(*props.get('offset_range', [0.2, 2.1])) # offset to previous element on conveyor
    scale_width = np.random.uniform(*props.get('scale_range_width', [0.7, 1.3]))
    scale_height = np.random.uniform(*props.get('scale_range_height', [0.7, 1.3]))
    scale_depth = np.random.uniform(*props.get('scale_range_depth', [0.7, 1.3]))
    rotation = np.random.uniform(*props.get('rotation_range', [-40, 40]))
    
    obj.scale = [scale_width, scale_depth, scale_height]
    obj.rotation_euler = Euler(np.radians([0.0, 0.0, rotation]), 'XYZ')
    obj.location = [conveyor_pos - off, 0, 0.05]

    LAYER.update()

    # Compute extension along conveyor direction, so that
    # subsequent objects do not overlap visually.
    bbox_min, bbox_max = compute_world_bbox(obj)
    ext = bbox_max[0] - bbox_min[0] 

    return conveyor_pos - (ext + off) 

def create_objects(conveyor, num_objects=10, **prop_kwargs):
    conveyor_pos=-2.0

    objs = []
    for i in range(num_objects):
        obj = create_object(conveyor)
        conveyor_pos = change_object_properties(obj, conveyor_pos, **prop_kwargs)
        objs += [obj]

    return objs, conveyor_pos

def create_conveyor(collection):
    # Add an emtpy that will act as parent to all generated objects
    bpy.ops.object.empty_add(type='ARROWS', location=(0,0,0))    
    conveyor = bpy.context.active_object
    conveyor.name = 'Conveyor'
    try:
        collection.objects.link(conveyor)
        SCN.collection.objects.unlink(conveyor) 
    except RuntimeError:
        pass
    return conveyor

def setup_animation(conveyor, conveyor_pos, **anim_props):
    
    # Setup conveyor speed and jiggle
    speed = anim_props.get("conveyor_speed", 0.1)
    drvForward = conveyor.driver_add('location', 0)
    drvForward.driver.expression = f'frame*{speed}'
    
    # For the jiggle we need to register a custom driver function
    jigglefac = anim_props.get("conveyor_jiggle_factor", 0.05)
    jiggle = lambda: np.random.random()*jigglefac
    bpy.app.driver_namespace["jiggle"] = jiggle
    
    drvJiggle = conveyor.driver_add('location', 2)
    drvJiggle.driver.expression = f'jiggle()'

    # Set animation range. TODO pytorch-blender does not
    # support different animation lengths.
    #SCN.frame_start=0
    #SCN.frame_end = int(abs(conveyor_pos)/speed)

def create_scene(num_objects=10, **kwargs):
    # Create collection to place generate generated objects into
    gcoll = SCN.collection.children['Generated']
    conveyor = create_conveyor(gcoll)
    objs, conveyor_pos = create_objects(conveyor, **kwargs)
    setup_animation(conveyor, conveyor_pos, **kwargs)
    return objs

def remove_objects():
    coll = SCN.collection.children['Generated']
    for o in coll.objects:
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.outliner.orphans_purge()