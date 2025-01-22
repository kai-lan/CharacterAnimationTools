"""Create mirror motions"""
import os
import pandas as pd
from tqdm import tqdm
from anim import bvh

index_path = '../HumanML3D/index.csv'
save_path = '/mnt/d/Mocap/amass_data_BVH_SMPL'

index_file = pd.read_csv(index_path)
total_amount = index_file.shape[0]

for i in tqdm(range(27, total_amount)):
    source_path = index_file.loc[i]['source_path']
    name = index_file.loc[i]['new_name']
    old_name = save_path + '/'+ name.replace('.npy', '.bvh')
    if os.path.exists(old_name):
        anim = bvh.load(old_name)
        m_anim = anim.mirror()
        new_name = save_path + '/'+ 'M' + name.replace('.npy', '.bvh')
        # new_name = "/mnt/d/TMP/test.bvh"
        bvh.save(new_name, m_anim)

    else:
        print("Not found", old_name)
