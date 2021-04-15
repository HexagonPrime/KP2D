import os

scene_images_dir_list = []
for scene in os.listdir("/cluster/scratch/shecai/hypersim"):
    scene_dir = os.path.join("/cluster/scratch/shecai/hypersim", scene)
    scene_images_dir_list.append(os.path.join(scene_dir, "/images"))
print(scene_images_dir_list)