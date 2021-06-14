# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Adapted from: https://github.com/rpautrat/SuperPoint/blob/master/superpoint/evaluations/detector_evaluation.py

import random
from glob import glob
from os import path as osp

import cv2
import numpy as np

from kp2d.utils.keypoints import warp_keypoints_frame2frame

from torch import neg

from kp2d.datasets.data_loader import DataLoader


def compute_repeatability(data, reprojection, keep_k_points=300, distance_thresh=3, center_crop=False, visualize=False):
    """
    Compute the repeatability metric between 2 sets of keypoints inside data.

    Parameters
    ----------
    data: dict
        Input dictionary containing:
        image_shape: tuple (H,W)
            Original image shape.
        homography: numpy.ndarray (3,3)
            Ground truth homography.
        prob: numpy.ndarray (N,3)
            Keypoint vector, consisting of (x,y,probability).
        warped_prob: numpy.ndarray (N,3)
            Warped keypoint vector, consisting of (x,y,probability).
        metainfo, sourceframe, targetframe, scenter2tcenter, metainfo_inv:
            For reprojection ground truth.
    keep_k_points: int
        Number of keypoints to select, based on probability.
    distance_thresh: int
        Distance threshold in pixels for a corresponding keypoint to be considered a correct match.

    Returns
    -------    
    N1: int
        Number of true keypoints in the first image.
    N2: int
        Number of true keypoints in the second image.
    repeatability: float
        Keypoint repeatability metric.
    loc_err: float
        Keypoint localization error.
    """
    def filter_keypoints(points, shape, inliers):
        """ Keep only the points whose coordinates are inside the dimensions of shape and inside projection inliers. """
        mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) &\
               (points[:, 1] >= 0) & (points[:, 1] < shape[1])
        mask = mask & inliers
        return points[mask, :]

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their probability).
        points has shape (num_points, 3) where the last coordinate is the probability. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    # H = data['homography']
    shape = data['image_shape']
    metainfo, source_frame, target_frame, scenter2tcenter, metainfo_inv = \
        data['metainfo'], data['source_frame'], data['target_frame'], data['scenter2tcenter'], data['metainfo_inv']

    # Filter out predictions
    keypoints = data['prob'][:,:2]
    warped_keypoints = data['warped_prob']
    # Original: warped_keypoints = keep_true_keypoints(warped_keypoints, np.linalg.inv(H), shape)
    _, inliers_inv = warp_keypoints_frame2frame(
        warped_keypoints[:,:2], metainfo_inv, target_frame, source_frame, [neg(scenter2tcenter[0]), neg(scenter2tcenter[1])], reprojection)
    warped_keypoints = filter_keypoints(warped_keypoints, shape, inliers_inv)

    # Warp the original keypoints with the true homography
    true_warped_keypoints, inliers = warp_keypoints_frame2frame(
        keypoints, metainfo, source_frame, target_frame, scenter2tcenter, reprojection)
    true_warped_keypoints = np.stack([true_warped_keypoints[:, 0], true_warped_keypoints[:, 1], data['prob'][:, 2]], axis=-1)
    true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape, inliers)

    # if none of the keypoints in in view, the two frames is extremely likely to have no overlapping area
    if true_warped_keypoints.shape[0] < 1 and warped_keypoints.shape[0] < 1:
        return None, None, None, None

    # Keep only the keep_k_points best predictions
    warped_keypoints = select_k_best(warped_keypoints, keep_k_points)
    true_warped_keypoints = select_k_best(true_warped_keypoints, keep_k_points)
    keypoints = filter_keypoints(data['prob'], shape, inliers)
    keypoints = select_k_best(keypoints, keep_k_points)
    # for visualization
    points0 = keypoints.copy()
    points1 = warped_keypoints.copy()

    # Compute the repeatability
    N1 = true_warped_keypoints.shape[0]
    N2 = warped_keypoints.shape[0]
    true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
    warped_keypoints = np.expand_dims(warped_keypoints, 0)
    # shapes are broadcasted to N1 x N2 x 2:
    norm = np.linalg.norm(true_warped_keypoints - warped_keypoints, ord=None, axis=2)
    count1 = 0
    count2 = 0
    le1 = 0
    le2 = 0
    if N2 != 0:
        min1 = np.min(norm, axis=1)
        argmin1 = np.argmin(norm, axis=1)
        correct1 = (min1 <= distance_thresh)
        count1 = np.sum(correct1)
        le1 = min1[correct1].sum()
    if N1 != 0:
        min2 = np.min(norm, axis=0)
        argmin2 = np.argmin(norm, axis=0)
        correct2 = (min2 <= distance_thresh)
        count2 = np.sum(correct2)
        le2 = min2[correct2].sum()
    if N1 + N2 > 0 and count1 + count2 > 0:
        repeatability = (count1 + count2) / (N1 + N2)
        loc_err = (le1 + le2) / (count1 + count2)
    else:
        repeatability = -1
        loc_err = -1
    # visualization
    # dataloader = DataLoader('/scratch-second/shecai/hypersim/', image_type='jpg', verbose=False)
    # image_source = dataloader.loadBgr(source_frame[0], source_frame[1], source_frame[2], source_frame[3])
    # image_target = dataloader.loadBgr(target_frame[0], target_frame[1], target_frame[2], target_frame[3])
        
    # # Draw pixels.
    # green = (0, 255, 0)
    # red = (0, 0, 255)
    # size = 2
    # vis = np.concatenate((image_source, image_target), axis=1)
    # for px_s, px_t, inlier in zip(points0, points1[argmin1], correct1):
    #     px_s = [int(i) for i in px_s]
    #     px_t = [int(i) for i in px_t]
    #     px_t[0] = px_t[0] + 1024
    #     if inlier:
    #         c = (random.uniform(0, 1)*255, random.uniform(0, 1)*255, 0)
    #         cv2.circle(vis, px_s, 2, c)
    #         cv2.circle(vis, px_t, 2, c)
    #         # vis = cv2.line(vis, px_s, px_t, green, 1)
    #     else:
    #         cv2.circle(vis, px_s, 2, red)
    #         cv2.circle(vis, px_t, 2, red)
    #         # vis = cv2.line(vis, px_s, px_t, red, 1)
    # cv2.imwrite('vis.jpg', vis)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return N1, N2, repeatability, loc_err
