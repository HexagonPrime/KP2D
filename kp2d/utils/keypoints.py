# Copyright 2020 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
import torch


def warp_keypoints(keypoints, H):
    """Warp keypoints given a homography

    Parameters
    ----------
    keypoints: numpy.ndarray (N,2)
        Keypoint vector.
    H: numpy.ndarray (3,3)
        Homography.

    Returns
    -------
    warped_keypoints: numpy.ndarray (N,2)
        Warped keypoints vector.
    """
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))], axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H))
    return warped_points[:, :2] / warped_points[:, 2:]

def warp_keypoints_frame2frame(keypoints, metainfo, source_frame, target_frame, scenter2tcenter, projection):
    """Warp keypoints given a homography

    Parameters
    ----------
    keypoints: numpy.ndarray (N,2)
        Keypoint vector.
    metainfo, source_frame, target_frame, scenter2tcenter, projection, center_crop
        Needed for hypersim projection.

    Returns
    -------
    warped_keypoints: numpy.ndarray (N,2)
        Warped keypoints vector.
    """
    keypoints = np.copy(keypoints)
    source_position_map, target_position_map, source_reflectance_map, target_R_CW, target_t_CW = \
            metainfo[0][0], metainfo[1][0], metainfo[2][0], metainfo[3][0], metainfo[4][0]

    # the images were center cropped, therefore convert the keypoint locations to (1024, 768)
    keypoints = keypoints.transpose((1, 0))
    keypoints = torch.from_numpy(keypoints)
    _, px_target, inliers = projection.warp(keypoints,
                                            source_position_map,
                                            target_R_CW, target_t_CW,
                                            mask_fov=True,
                                            mask_occlusion=target_position_map,
                                            mask_reflectance=source_reflectance_map)
    px_target = px_target.permute(1, 0)
    return px_target.detach().cpu().clone().numpy(), inliers.detach().cpu().clone().numpy()

def draw_keypoints(img_l, top_uvz, color=(255, 0, 0), idx=0):
    """Draw keypoints on an image"""
    vis_xyd = top_uvz.permute(0, 2, 1)[idx].detach().cpu().clone().numpy()
    vis = img_l.copy()
    cnt = 0
    for pt in vis_xyd[:,:2].astype(np.int32):
        x, y = int(pt[0]), int(pt[1])
        # cv2.circle(vis, (x,y), 2, color, -1)
        cv2.circle(vis, (x,y), 2, color)
    return vis
