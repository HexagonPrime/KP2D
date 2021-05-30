import os
import pandas as pd
from tqdm import tqdm

images_dict_list = []
for scene in os.listdir("/scratch_net/biwidl306_second/shecai/hypersim"):
    scene_dir = os.path.join("/scratch_net/biwidl306_second/shecai/hypersim", scene)
    scene_images_dir = os.path.join(scene_dir, "images")
    for directory in os.listdir(scene_images_dir):
        if directory.endswith("_final_hdf5"):
            cam_trajectory_dir = os.path.join(scene_images_dir, directory)
            for image in os.listdir(cam_trajectory_dir):
                if image.endswith('.color.hdf5'):
                    image_path = os.path.join(cam_trajectory_dir, image)
                    images_dict_list.append([scene, directory, image_path])
df = pd.DataFrame(images_dict_list)
df.to_csv(os.path.join("/scratch_net/biwidl306_second/shecai/hypersim", 'hypersim.csv'), index=False, header=False)

# col_list = ["volume", "scene", "cam", "frame", "img_path"]
col_list = ['scene', 'cam', 'img_path']
df = pd.read_csv("/scratch_net/biwidl306_second/shecai/hypersim/hypersim.csv", names=col_list)
info_list = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    vol = int(row['img_path'][49:52])
    scene = int(row['img_path'][53:56])
    cam = int(row['img_path'][74:76])
    frame = int(row['img_path'][94:98])
    this_row = [vol, scene, cam, frame, row['img_path']]
    info_list.append(this_row)
df = pd.DataFrame(info_list)
df.to_csv(os.path.join("/scratch_net/biwidl306_second/shecai/hypersim", 'hypersim_0.csv'), index=False, header=False)