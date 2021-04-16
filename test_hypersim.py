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
    # print(filename)
    img_array = np.array(h5['dataset'][:], dtype='f')
    img_array = img_array * 255
    img_array = img_array.astype(np.uint8)
    # print(img_array.shape)
    img = Image.fromarray(img_array)
    img = img.resize((int(img.size[0]/3), int(img.size[1]/3)))
    # print (img.size)
    if not (img.size[0] == 341 and img.size[1] == 256):
        print("false")