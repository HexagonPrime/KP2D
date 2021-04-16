import os
import pandas as pd
from PIL import Image
import h5py
import numpy as np

col_list = ["scene", "cam_trajectory", "img_path"]
df = pd.read_csv('/cluster/scratch/shecai/hypersim/hypersim.csv', names=col_list)
files = df["img_path"].tolist()
for filename in files:
    h5 = h5py.File(filename, 'r')
    img_array = np.array(h5['dataset'][:], dtype='f')
    print(img_array.shape)
    Image.fromarray(img_array)