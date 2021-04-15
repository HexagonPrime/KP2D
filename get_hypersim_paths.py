import os

images_dict_list[]
for scene in os.listdir("/cluster/scratch/shecai/hypersim"):
    scene_dir = os.path.join("/cluster/scratch/shecai/hypersim", scene)
    print (scene_dir)
    scene_images_dir = os.path.join(scene_dir, "images")
    print (scene_images_dir)
    # scene_images_dir_list.append(scene_images_dir)
    for directory in os.listdir(scene_images_dir):
        if directory.endswith("_final_hdf5"):
            cam_trajectory_dir = os.path.join(scene_images_dir, directory)
            for image in os.listdir(cam_trajectory_dir)
                image_path = os.path.join(cam_trajectory_dir, image)
                images_dict_list.append([scene, cam_trajectory_dir, image_path])
print (images_dict_list)

# print(scene_images_dir_list)

# for scene_images_dir in scene_images_dir_list:
#     for directory in os.listdir(scene_images_dir):
#         if directory.endswith("_final_hdf5"):
