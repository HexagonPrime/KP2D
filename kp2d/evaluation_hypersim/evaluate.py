# Copyright 2020 Toyota Research Institute.  All rights reserved.

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# from kp2d.evaluation_hypersim.descriptor_evaluation import (compute_homography,
#                                                    compute_matching_score)
from kp2d.evaluation_hypersim.descriptor_evaluation import compute_matching_score
from kp2d.evaluation_hypersim.detector_evaluation import compute_repeatability
from kp2d.utils.image import to_color_normalized, to_gray_normalized
from kp2d.utils.reprojection import Reprojection
from math import isnan


def evaluate_keypoint_net_hypersim(data_loader, keypoint_net, output_shape=(1024, 768), top_k=1000, use_color=True, center_crop=False):
    """Keypoint net evaluation script. 

    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        Dataset loader. 
    keypoint_net: torch.nn.module
        Keypoint network.
    output_shape: tuple
        Original image shape.
    top_k: int
        Number of keypoints to use to compute metrics, selected based on probability.    
    use_color: bool
        Use color or grayscale images.
    """
    reprojection = Reprojection(width=1024, height=768, verbose=False)
    keypoint_net.eval()
    keypoint_net.training = False

    conf_threshold = 0.0
    localization_err, repeatability = [], []
    MScore = []
    # correctness1, correctness3, correctness5, MScore = [], [], [], []

    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="evaluate_keypoint_net"):
            if use_color:
                image = to_color_normalized(sample['image_source'].cuda())
                warped_image = to_color_normalized(sample['image_target'].cuda())
            else:
                image = to_gray_normalized(sample['image_source'].cuda())
                warped_image = to_gray_normalized(sample['image_target'].cuda())

            score_1, coord_1, desc1 = keypoint_net(image)
            score_2, coord_2, desc2 = keypoint_net(warped_image)
            B, C, Hc, Wc = desc1.shape

            # Scores & Descriptors
            score_1 = torch.cat([coord_1, score_1], dim=1).view(3, -1).t().cpu().numpy()
            score_2 = torch.cat([coord_2, score_2], dim=1).view(3, -1).t().cpu().numpy()
            desc1 = desc1.view(C, Hc, Wc).view(C, -1).t().cpu().numpy()
            desc2 = desc2.view(C, Hc, Wc).view(C, -1).t().cpu().numpy()
            
            # Filter based on confidence threshold
            desc1 = desc1[score_1[:, 2] > conf_threshold, :]
            desc2 = desc2[score_2[:, 2] > conf_threshold, :]
            score_1 = score_1[score_1[:, 2] > conf_threshold, :]
            score_2 = score_2[score_2[:, 2] > conf_threshold, :]

            # Prepare data for eval
            data = {'image': sample['image_source'].numpy().squeeze(),
                    'image_shape' : output_shape,
                    'warped_image': sample['image_target'].numpy().squeeze(),
                    # 'homography': sample['homography'].squeeze().numpy(),
                    'metainfo': sample['metainfo'],
                    'source_frame': sample['source_frame'],
                    'target_frame': sample['target_frame'],
                    'scenter2tcenter': sample['scenter2tcenter'],
                    'metainfo_inv': sample['metainfo_inv'],
                    'prob': score_1, 
                    'warped_prob': score_2,
                    'desc': desc1,
                    'warped_desc': desc2}
            
            # Compute repeatabilty and localization error
            N1, N2, rep, loc_err = compute_repeatability(data, reprojection, keep_k_points=top_k, distance_thresh=3, center_crop=center_crop)
            if not loc_err == None:
                repeatability.append(rep)
                localization_err.append(loc_err)

                # Compute correctness
                # c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
                # correctness1.append(c1)
                # correctness3.append(c2)
                # correctness5.append(c3)

                # Compute matching score
                mscore = compute_matching_score(data, reprojection, keep_k_points=top_k, center_crop=center_crop)
                MScore.append(mscore)
            # else:
                # print(str(N1) + ' ' + str(N2) + ' ' + str(rep) + ' ' + str(loc_err))

    # return np.mean(repeatability), np.mean(localization_err), \
    #        np.mean(correctness1), np.mean(correctness3), np.mean(correctness5), np.mean(MScore)
    # return np.mean(repeatability), np.mean(localization_err)
    return np.mean(repeatability), np.mean(localization_err), np.mean(MScore)
