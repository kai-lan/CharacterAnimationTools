import sys, os
sys.path.append("joints2bvh")
import numpy as np
from tqdm import tqdm
import pandas as pd
from anim import bvh, amass
from joints2bvh.joint2bvh import Joint2BVHConvertor

"""Look through the index file and save to BVH (scale by 100 for motionbuilder)"""


template_file = "/mnt/d/Mocap/amass_data_BVH_scaled/000021.bvh" # run IK for humanact12
index_path = '../HumanML3D/index.csv'
fps = 20 # same as HumanML3D
save_path = '/mnt/d/Mocap/amass_data_BVH_scaled'
os.makedirs(save_path, exist_ok=True)

paths = []
folders = []
dataset_names = []
for root, dirs, files in os.walk('/mnt/d/Mocap/amass_data'):
    folders.append(root)
    for name in files:
        if name.endswith('.txt'): continue
        dataset_name = root.split('/')[2]
        if dataset_name not in dataset_names:
            dataset_names.append(dataset_name)
        paths.append(os.path.join(root, name))

group_path = [[path for path in paths if name in path] for name in dataset_names]
paths = [path for gp in group_path for path in gp]

index_file = pd.read_csv(index_path)
total_amount = index_file.shape[0]

converter = Joint2BVHConvertor(template_file)
scale = 100.0 # Motionbuilder scale

for i in tqdm(range(total_amount)):
    source_path = index_file.loc[i]['source_path']
    new_name = save_path + '/'+ index_file.loc[i]['new_name'].replace('.npy', '.bvh')
    if os.path.exists(new_name):
        continue

    if 'humanact12' in source_path:
        source_path = source_path.replace('./pose_data', '../HumanML3D/pose_data')
        joints = np.load(source_path)[:, :22]
        joints *= scale
        """Put them on the ground"""
        min_pos = joints[..., 1].min()
        joints[..., 1] -= min_pos
        converter.convert(joints, new_name, foot_ik=False)
    else:
        continue
        source_path = source_path.replace('./pose_data', '/mnt/d/Mocap/amass_data').replace('.npy', '.npz')

        anim = amass.load(
            amass_motion_path=source_path,
            smplh_path="../HumanML3D/body_models/smplh",
            load_hand=False,
            remove_betas=False,
            scale=scale,
            gender=None,
            num_joints=22
        )

        bvh.save(
            filepath=new_name,
            anim=anim
        )


