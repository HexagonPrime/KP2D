from kp2d.datasets.data_loader import DataLoader
from kp2d.utils.reprojection_for_testing import Reprojection
from PIL import Image
from torchvision.transforms.functional import crop

import numpy as np
import cv2
import torch
import os
import time

# Initialize utilities once.
verbose = False
all_pixels = True
# Assume hypersim data sits in $SCRATCH/hypersim
#hypersim_path = os.environ.get('SCRATCH')
#hypersim_path = os.path.join(hypersim_path, 'hypersim')
hypersim_path = "/scratch_net/biwidl306_second/shecai/hypersim/"
data = DataLoader(hypersim_path, image_type='jpg', verbose=verbose)
reprojection = Reprojection(width=1024, height=768, verbose=verbose)

# Load data. This will be cached in data loader.
vol=4
scene=1
cam=0
source_frame=0
target_frame=1
source_position_map = data.loadPositionMap(vol, scene, cam, source_frame)
target_position_map = data.loadPositionMap(vol, scene, cam, target_frame)
source_reflectance_map = data.loadReflectance(vol, scene, cam, source_frame)
R_CW, t_CW = data.loadCamPose(vol, scene, cam, target_frame)
source_center = torch.tensor([[512],[384]])
_, target_center, inliers = reprojection.warp(source_center,
                                source_position_map,
                                R_CW, t_CW,
                                mask_fov=True,
                                mask_occlusion=None,
                                mask_reflectance=None)
target_center = target_center.long()
center2center = target_center - source_center
print(center2center)

# Warp test pixels.
# First column: width, second column height.
px_source = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800, 900],
                          [600, 600, 600, 600, 600, 600, 600, 600, 600]])
if all_pixels:
    import itertools
    W = [*range(0, 512, 4)]
    H = [*range(0, 384, 4)]
    px_source=torch.tensor(list(itertools.product(W, H))).T
print(px_source)
px_source[0] = px_source[0]+256
px_source[1] = px_source[1]+192

start_time = time.time()
px_source, px_target, inliers = reprojection.warp(px_source,
                                                source_position_map,
                                                R_CW, t_CW,
                                                mask_fov=True,
                                                mask_occlusion=target_position_map,
                                                mask_reflectance=source_reflectance_map)
print('Warping operation: {:.3f} seconds'.format(time.time() - start_time))
px_source[0] = px_source[0]-256
px_source[1] = px_source[1]-192
print(px_source)
px_target[0] = px_target[0]-256
px_target[1] = px_target[1]-192
px_target[0] = px_target[0] - center2center[0]
px_target[1] = px_target[1] - center2center[1]
print(px_target)

# Visualize
image_source = data.loadBgr(vol, scene, cam, source_frame)
image_target = data.loadBgr(vol, scene, cam, target_frame)
image_source = image_source.astype(np.uint8)
image_target = image_target.astype(np.uint8)
image_source = Image.fromarray(image_source)
image_target = Image.fromarray(image_target)
image_target = crop(image_target, 192+center2center[1].item(), 256+center2center[0].item(), 384, 512)
image_source = crop(image_source, 192, 256, 384, 512)
image_target = np.array(image_target)
image_source = np.array(image_source) 
cv2.imwrite('image_source_ori.jpg', image_source)
cv2.imwrite('image_target_ori.jpg', image_target)

# Draw pixels.
blue = (255, 0, 0)
red = (0, 0, 255)
size = 2
px_source = px_source.cpu().detach().numpy()
px_target = px_target.cpu().detach().numpy()
inliers = inliers.cpu().detach().numpy()
for px_s, px_t, inlier in zip(px_source.T, px_target.T, inliers):
    # print(px_s.type)
    # print(px_t.type)
    # print(inlier.type)
    if inlier:
        cv2.circle(image_source, tuple(px_s), size, blue)
        cv2.circle(image_target, tuple(px_t), size, blue)
    else:
        cv2.circle(image_source, tuple(px_s), size, red)
        cv2.circle(image_target, tuple(px_t), size, red)

# cv2.imshow('image_source', img_source)
# cv2.imshow('image_target', img_target)
cv2.imwrite('image_source.jpg', image_source)
cv2.imwrite('image_target.jpg', image_target)
cv2.waitKey(0)
cv2.destroyAllWindows()