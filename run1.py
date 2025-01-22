""" Use FK to retarget a BVH animation to a uniform skeleton.
    Only scale root position, and keep the rotations.
    Choose 000004.bvh as the template offsets.
"""
import os
import pandas as pd
from tqdm import tqdm
from anim import bvh
from anim import skel

index_path = '../HumanML3D/index.csv'
old_path = '/mnt/d/Mocap/amass_data_BVH_scaled'
save_path = '/mnt/d/Mocap/amass_data_BVH_SMPL'
template_file = "/mnt/d/Mocap/amass_data_BVH_scaled/000004.bvh"

os.makedirs(save_path, exist_ok=True)

index_file = pd.read_csv(index_path)
total_amount = index_file.shape[0]

# Load template offsets
template_anim = bvh.load(template_file)
template_offsets = template_anim.offsets
template_height = template_offsets[:, 1].max() - template_offsets[:, 1].min()

# for i in tqdm(range(total_amount)):
for i in range(27, 28):
    source_path = index_file.loc[i]['source_path']
    old_name = old_path + '/'+ index_file.loc[i]['new_name'].replace('.npy', '.bvh')
    new_name = save_path + '/'+ index_file.loc[i]['new_name'].replace('.npy', '.bvh')
    if os.path.exists(old_name):
        anim = bvh.load(old_name)
        offsets = anim.offsets
        height = offsets[:, 1].max() - offsets[:, 1].min()
        # Scale root position by height
        anim.trans[..., [0, 2]] *= template_height / height

        anim.skel = skel.Skel.from_names_parents_offsets(
            names=anim.skel.names,
            parents=anim.skel.parents,
            offsets=template_offsets,
            skel_name=anim.skel.skel_name
        )

        # Put on the floor
        anim.trans[..., 1] -= anim.gpos[..., 1].min()
        bvh.save(new_name, anim)
        # exit()
    else:
        print("Not found", old_name)
