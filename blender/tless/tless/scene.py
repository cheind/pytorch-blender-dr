import bpy
import bmesh
import numpy as np
import supershape as sshape
from blendtorch import btb
from .config import DEFAULT_CONFIG

SCN = bpy.context.scene
LAYER = bpy.context.view_layer

OCC_SHAPE = (50,50)
def create_occluder_template():
    shape=OCC_SHAPE
    new_obj = sshape.make_bpy_mesh(shape, name='occ', coll=False)
    new_obj.use_fake_user = True
    return new_obj
OCC = create_occluder_template()

def randomize_bsdf_material(mat, basecolor=None):
    bsdf = mat.node_tree.nodes.get("Principled BSDF") 
    if basecolor is None:
        basecolor = np.random.uniform(0,1,size=3)
    bsdf.inputs["Base Color"].default_value = np.concatenate((basecolor, [1.]))
    return mat

def create_bsdf_material(basecolor=None):
    mat = bpy.data.materials.new("randommat")
    mat.use_nodes = True
    randomize_bsdf_material(mat, basecolor=basecolor)
    return mat

def randomize_box_material(cfg):
    box = bpy.data.objects['Box']
    if cfg['scene.background_material'] == 'magic':
        mat = bpy.data.materials['BoxMaterialMagic']
        node_tree = mat.node_tree
        nodes = node_tree.nodes
        mapping = nodes.get('Mapping')
        mapping.inputs['Rotation'].default_value = np.random.uniform(-1.0, 1.0, size=3)
        mapping.inputs['Scale'].default_value = np.random.uniform(0.1, 3.0, size=3)
    elif cfg['scene.background_material'] == 'plain':
        mat = create_bsdf_material()
        randomize_bsdf_material(mat)
    # Box already has materials, so assign to slot 0 instead of active_material.
    box.data.materials[0] = mat

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
    #new_obj.data = templates[c].data.copy()
    #new_obj.modifiers.clear()
    #new_obj.constraints.clear()
    new_obj.animation_data_clear()
    
    intensity = np.random.uniform(*cfg['scene.object_intensity_range'])
    new_obj.active_material = create_bsdf_material((intensity, intensity, intensity))
    gcoll.objects.link(new_obj)
    
    new_obj.location = np.random.uniform(low=cfg['scene.object_location_bbox'][0],high=cfg['scene.object_location_bbox'][1],size=(3))
    new_obj.rotation_euler = np.random.uniform(low=cfg['scene.object_rotation_range'][0], high=cfg['scene.object_rotation_range'][1],size=(3))
    try:
        SCN.rigidbody_world.collection.objects.link(new_obj)
    except:
        pass
    return new_obj, c
    
def create_occluder(cfg=DEFAULT_CONFIG):
    coll = SCN.collection.children['Occluders']

    new_obj = OCC.copy()
    new_obj.data = OCC.data.copy()
    new_obj.active_material = create_bsdf_material()
    coll.objects.link(new_obj)
        
    params = np.random.uniform(
        low =[0.00,1,1,0.0,0.0, 0.0],
        high=[20.00,1,1,40,10.0,10.0],
        size=(2,6)
    )
    
    scale = np.random.uniform(0.1, 0.6, size=3)    
    x,y,z = sshape.supercoords(params, shape=OCC_SHAPE)    
    sshape.update_bpy_mesh(x*scale[0], y*scale[1], z*scale[2], new_obj)
    
    new_obj.location = np.random.uniform(low=[-2, -2, 3],high=[2,2,6],size=(3))
    new_obj.rotation_euler = np.random.uniform(low=-np.pi, high=np.pi,size=(3))
    SCN.rigidbody_world.collection.objects.link(new_obj)
    
    return new_obj

def remove_objects(objs, occs):
    coll = SCN.collection.children['Generated']

    mats = []
    for (o,c) in objs:
        o.data.materials.clear()
        SCN.rigidbody_world.collection.objects.unlink(o)                
        bpy.data.objects.remove(o, do_unlink=True)

    for o in occs:
        o.data.materials.clear()
        SCN.rigidbody_world.collection.objects.unlink(o) 
        bpy.data.objects.remove(o, do_unlink=True)
    
    for m in list(bpy.data.materials):
        if m.users == 0:
            bpy.data.materials.remove(m, do_unlink=True)          

    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m, do_unlink=True)    
    #bpy.ops.outliner.orphans_purge()

def apply_physics_to(objs, enabled=False, collision_shape='BOX', friction=0.5, linear_damp=0.05, angular_damp=0.1):
    for obj in objs:
        obj.rigid_body.enabled = enabled
        obj.rigid_body.collision_shape = collision_shape
        obj.rigid_body.friction = friction
        obj.rigid_body.linear_damping = linear_damp
        obj.rigid_body.angular_damping = angular_damp

def create_scene(cfg=DEFAULT_CONFIG):    
    N = cfg['scene.num_objects']
    M = np.random.binomial(N, cfg['scene.prob_occluder'],size=1).sum()

    objs = [create_object(cfg) for _ in range(N)]
    occs = [create_occluder(cfg) for _ in range(M)]

    
    apply_physics_to(
        [o[0] for o in objs],
        collision_shape='BOX',
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
        
    randomize_box_material(cfg)
    
    # Use real background images on plane (and make box transparent)
    bgImgFolder = cfg['scene.background_images']
    if (len(bgImgFolder) > 0):
        setupTexturePlane(bgImgFolder)
        bpy.data.objects['Plane'].parent = bpy.data.objects['Camera']
        
    return objs, occs

def simplified_templates(num_target_faces=200):
    tcoll = SCN.collection.children['Objects']
    gcoll = SCN.collection.children['Generated']

    templates = list(tcoll.objects)
    simplified = []
    bm = bmesh.new()
    for t in templates:

        bm.from_mesh(t.data)
        res = bmesh.ops.convex_hull(bm, input=bm.verts)
        bmesh.ops.delete(bm, geom=res['geom_interior'], context='VERTS') 
        bmesh.ops.delete(bm, geom=res['geom_unused'], context='VERTS')
        bmesh.ops.delete(bm, geom=res['geom_holes'], context='VERTS')
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.01)
        bmesh.ops.join_triangles(
            bm, faces=bm.faces, cmp_seam=False, cmp_sharp=False, 
            cmp_uvs=False, cmp_vcols=False, cmp_materials=False, 
            angle_face_threshold=0.698132, angle_shape_threshold=0.698132)

        ch = bpy.data.meshes.new("%s convexhull" % t.name)
        bm.to_mesh(ch)
        o = t.copy()
        o.data = ch        
        mod = o.modifiers.new('DecimateMod','DECIMATE')
        print(len(o.data.polygons), num_target_faces/len(o.data.polygons))
        mod.decimate_type='COLLAPSE'
        mod.ratio = num_target_faces/len(o.data.polygons)
        mod.use_symmetry = False
        mod.use_collapse_triangulate = True        
        gcoll.objects.link(o)
        simplified.append(o)
        bm.clear()
    bm.free()

    depsgraph = bpy.context.evaluated_depsgraph_get()
    all_xyz = []
    for s in simplified:
        s_eval = s.evaluated_get(depsgraph)
        xyz = btb.utils.object_coordinates(s_eval)
        all_xyz.append(xyz)
        print(len(s_eval.data.polygons), xyz.shape)
        del s_eval
        #bpy.data.meshes.remove(s.data)
        bpy.data.objects.remove(s, do_unlink=True)
    return all_xyz
    

def setTexture(objName, textureFileName):

    # get the object
    obj = bpy.data.objects[objName]

    # get or create material
    mat = bpy.data.materials.get("TextureMaterial_" + objName)
    if mat is None:
        # remove current materials from object
        while(len(obj.data.materials) > 0):
            obj.data.materials.pop()
        # create and assign new material
        mat = bpy.data.materials.new(name="TextureMaterial_" + objName)
        obj.data.materials.append(mat)
    else: 
        #print("texture material already exists!")
        pass

    # link the texture node directly to the material output node (without shading)
    mat.use_nodes = True
    
    #while(len(mat.node_tree.nodes) > 0):
    #    mat.node_tree.nodes.pop()
        
    for img in bpy.data.images: 
        bpy.data.images.remove(img)
    
    matOut = mat.node_tree.nodes['Material Output']
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    # most of the time use a real image for wrench texture, otherwise use a synthetic image
    texImage.image = bpy.data.images.load(textureFileName)
    mat.node_tree.links.new(matOut.inputs['Surface'], texImage.outputs['Color'])

def setupTexturePlane(bgImageFolder):

    # Make box transparent
    makeObjectTransparent('Box')
    
    # select random background image from directory of background images
    import os
    import random
    bgImageFileNames = os.listdir(bgImageFolder)
    randomIndex = random.choice(range(len(bgImageFileNames)))
    fileName = bgImageFolder + bgImageFileNames[randomIndex]
    print("loading image: " + fileName)

    # create a new plane object (if there is no one yet)
    if not 'Plane' in bpy.data.objects.keys():
        bpy.ops.mesh.primitive_plane_add(size=35, location=(0, 0, -30), rotation=(0, 0, 0))
        plane = bpy.data.objects['Plane']
        plane.scale[1] = 2.0 / 3.0
    
    # load a random image and assign as texture to the Plane
    
    setTexture('Plane', fileName)

def alignPlaneWithCamera():
    
    # Position the plane parallel to the imaging plane at some distance ahead of the camera
    from mathutils import Vector, Matrix
    plane = bpy.data.objects['Plane']
    m = Matrix(bpy.data.objects['Camera'].matrix_world)
    plane.matrix_world = m
    zVec = Vector([m[0][2], m[1][2], m[2][2]]) * -30
    plane.matrix_world[0][3] += zVec[0]
    plane.matrix_world[1][3] += zVec[1]
    plane.matrix_world[2][3] += zVec[2]

    # scale the Plane in Y by 2/3
    plane.scale[1] = 2.0 / 3.0

def makeObjectTransparent(objName = 'Box'):
    # Make box transparent
    bpy.data.objects[objName].data.materials[0].blend_method = 'CLIP'
    bpy.data.objects[objName].data.materials[0].shadow_method = 'CLIP'
    bpy.data.objects[objName].data.materials[0].alpha_threshold = 1.0

def setCameraBackgroundImage(bgImageFolder):
    import os
    import random
    import bpy
    
    # cfg = {'scene.background_images': 'd:/data/val2017/'}
    bgImages = bpy.data.objects['Camera'].data.background_images
    while (len(bgImages) > 0):
        bgImages.remove(bgImages[-1])
    for img in bpy.data.images: 
        bpy.data.images.remove(img)
    #
    # select random background image from directory of background images
    #bgImageFolder = cfg['scene.background_images']
    bgImageFileNames = os.listdir(bgImageFolder)
    randomIndex = random.choice(range(len(bgImageFileNames)))
    fileName = bgImageFolder + bgImageFileNames[randomIndex]
    print("loading image: " + fileName)
    
    #image = bpy.data.images.load(fileName)
    #bpy.ops.view3d.background_image_add(image)
        
    bpy.data.objects['Camera'].data.background_images.new()
    bpy.data.objects['Camera'].data.background_images[0].image = bpy.data.images.load(fileName)
    bpy.data.objects['Camera'].data.show_background_images = True

