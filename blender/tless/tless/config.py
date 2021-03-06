import json
from pathlib import Path
import numpy as np

# DEFAULT_CONFIG = {
#     'camera.num_images' : 4,
#     'camera.radius_range': (8,10),
#     'camera.theta_range': (0,np.pi/4),
#     'camera.lookat': (0.,0.,0.),
#     'scene.num_objects': 10,
#     'scene.prob_occluder': 0.5, # additional occluder probability per object
#     'scene.object_cls_prob': None, # None -> equal probs, otherwise dict(classid -> prob)
#     'scene.object_intensity_range': (0.1,1),
#     'scene.object_location_bbox': [[-2, -2, 1],[2,2,4]],
#     'scene.object_rotation_range': (-np.pi, np.pi),
#     'scene.background_material': 'magic', # 'magic' or 'plain'.
#     'physics.linear_damp': 0.5,
#     'physics.angular_damp': 0.5,
#     'physics.friction': 0.8,
# }
# with open(Path(__file__).parent/'config.json', 'w') as fp:
#     fp.write(json.dumps(DEFAULT_CONFIG, indent=4))

with open(Path(__file__).parent.parent/'config.json', 'r') as fp:
    DEFAULT_CONFIG = json.loads(fp.read())
