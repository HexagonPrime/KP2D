from PIL import Image
import pandas as pd
import h5py
import numpy as np
import cv2

col_list = ["scene", "cam_trajectory", "img_path"]
df = pd.read_csv('/cluster/scratch/shecai/hypersim/hypersim.csv', names=col_list)
files = df["img_path"].tolist()

for file in files:
    h5 = h5py.File(file, 'r')
    image = np.array(h5['dataset'][:], dtype='f')
    image[image>1] = 1
    image = image * 255
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.save(file[:-4] + 'jpg')