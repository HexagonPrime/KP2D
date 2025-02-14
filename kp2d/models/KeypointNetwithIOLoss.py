# Copyright 2020 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.cm import get_cmap

from kp2d.networks.inlier_net import InlierNet
from kp2d.networks.keypoint_net import KeypointNet
from kp2d.networks.keypoint_resnet import KeypointResnet
from kp2d.utils.image import (image_grid, to_color_normalized,
                              to_gray_normalized)
from kp2d.utils.keypoints import draw_keypoints

from kp2d.utils.reprojection import Reprojection
from tqdm import tqdm


def build_descriptor_loss(source_des, target_des, source_points, tar_points, tar_points_un, keypoint_mask=None, relax_field=8, eval_only=False):
    """Desc Head Loss, per-pixel level triplet loss from https://arxiv.org/pdf/1902.11046.pdf..
    Parameters
    ----------
    source_des: torch.Tensor (B,256,H/8,W/8)
        Source image descriptors.
    target_des: torch.Tensor (B,256,H/8,W/8)
        Target image descriptors.
    source_points: torch.Tensor (B,H/8,W/8,2)
        Source image keypoints
    tar_points: torch.Tensor (B,H/8,W/8,2)
        Target image keypoints
    tar_points_un: torch.Tensor (B,2,H/8,W/8)
        Target image keypoints unnormalized
    eval_only: bool
        Computes only recall without the loss.
    Returns
    -------
    loss: torch.Tensor
        Descriptor loss.
    recall: torch.Tensor
        Descriptor match recall.
    """
    device = source_des.device
    batch_size, C, _, _ = source_des.shape
    loss, recall = 0., 0.
    margins = 0.2

    for cur_ind in range(batch_size):

        if keypoint_mask is None:
            ref_desc = torch.nn.functional.grid_sample(source_des[cur_ind].unsqueeze(0), source_points[cur_ind].unsqueeze(0), align_corners=True).squeeze().view(C, -1)
            tar_desc = torch.nn.functional.grid_sample(target_des[cur_ind].unsqueeze(0), tar_points[cur_ind].unsqueeze(0), align_corners=True).squeeze().view(C, -1)
            tar_points_raw = tar_points_un[cur_ind].view(2, -1)
        else:
            keypoint_mask_ind = keypoint_mask[cur_ind].squeeze()

            n_feat = keypoint_mask_ind.sum().item()
            if n_feat < 20:
                continue

            ref_desc = torch.nn.functional.grid_sample(source_des[cur_ind].unsqueeze(0), source_points[cur_ind].unsqueeze(0), align_corners=True).squeeze()[:, keypoint_mask_ind]
            tar_desc = torch.nn.functional.grid_sample(target_des[cur_ind].unsqueeze(0), tar_points[cur_ind].unsqueeze(0), align_corners=True).squeeze()[:, keypoint_mask_ind]
            tar_points_raw = tar_points_un[cur_ind][:, keypoint_mask_ind]

        # Compute dense descriptor distance matrix and find nearest neighbor
        ref_desc = ref_desc.div(torch.norm(ref_desc, p=2, dim=0))
        tar_desc = tar_desc.div(torch.norm(tar_desc, p=2, dim=0))
        dmat = torch.mm(ref_desc.t(), tar_desc)
        dmat = torch.sqrt(2 - 2 * torch.clamp(dmat, min=-1, max=1))

        # Sort distance matrix
        dmat_sorted, idx = torch.sort(dmat, dim=1)

        # Compute triplet loss and recall
        candidates = idx.t() # Candidates, sorted by descriptor distance

        # Get corresponding keypoint positions for each candidate descriptor
        match_k_x = tar_points_raw[0, candidates]
        match_k_y = tar_points_raw[1, candidates]

        # True keypoint coordinates
        true_x = tar_points_raw[0]
        true_y = tar_points_raw[1]

        # Compute recall as the number of correct matches, i.e. the first match is the correct one
        correct_matches = (abs(match_k_x[0]-true_x) == 0) & (abs(match_k_y[0]-true_y) == 0)
        recall += float(1.0 / batch_size) * (float(correct_matches.float().sum()) / float( ref_desc.size(1)))

        if eval_only:
            continue

        # Compute correct matches, allowing for a few pixels tolerance (i.e. relax_field)
        correct_idx = (abs(match_k_x - true_x) <= relax_field) & (abs(match_k_y - true_y) <= relax_field)
        # Get hardest negative example as an incorrect match and with the smallest descriptor distance 
        incorrect_first = dmat_sorted.t()
        incorrect_first[correct_idx] = 2.0 # largest distance is at most 2
        incorrect_first = torch.argmin(incorrect_first, dim=0)
        incorrect_first_index = candidates.gather(0, incorrect_first.unsqueeze(0)).squeeze()

        anchor_var = ref_desc
        pos_var    = tar_desc
        neg_var    = tar_desc[:, incorrect_first_index]

        loss += float(1.0 / batch_size) * torch.nn.functional.triplet_margin_loss(anchor_var.t(), pos_var.t(), neg_var.t(), margin=margins)

    return loss, recall


def warp_homography_batch(sources, homographies):
    """Batch warp keypoints given homographies.

    Parameters
    ----------
    sources: torch.Tensor (B,H,W,C)
        Keypoints vector.
    homographies: torch.Tensor (B,3,3)
        Homographies.

    Returns
    -------
    warped_sources: torch.Tensor (B,H,W,C)
        Warped keypoints vector.
    """
    B, H, W, _ = sources.shape
    warped_sources = []
    for b in range(B):
        source = sources[b].clone()
        source = source.view(-1,2)
        source = torch.addmm(homographies[b,:,2], source, homographies[b,:,:2].t())
        source.mul_(1/source[:,2].unsqueeze(1))
        source = source[:,:2].contiguous().view(H,W,2)
        warped_sources.append(source)
    return torch.stack(warped_sources, dim=0)

def warp_frame2frame_batch(sources, metainfo, source_frame, target_frame, scenter2tcenter, projection):
    """Batch warp keypoints given homographies.

    Parameters
    ----------
    sources: torch.Tensor [B, 2, 48, 64]
        Keypoints vector.
    metainfosource_frame, target_frame, scenter2tcenter, projection
        Information needed to warp keypoint locations from source frame to target frame.

    Returns
    -------
    warped_sources: torch.Tensor (B,H,W,C)
        Warped keypoints vector.
    """
    B, C, H, W = sources.shape
    warped_sources = []
    inliers_list = []
    source_name_list = []
    target_name_list = []
    for b in range(B):
        source = sources[b].clone()
        source_name = str(source_frame[0][b].item()) \
                    + '_' + str(source_frame[1][b].item()) \
                    + '_' + str(source_frame[2][b].item()) \
                    + '_' + str(source_frame[3][b].item())
        source_position_map, target_position_map, source_reflectance_map, target_R_CW, target_t_CW = \
                metainfo[0][b], metainfo[1][b], metainfo[2][b], metainfo[3][b], metainfo[4][b]
        this_scenter2tcenter = [scenter2tcenter[0][b], scenter2tcenter[1][b]]
        source = source.view(C, H*W)
        # the images were center cropped, therefore convert the keypoint locations to (1024, 768)
        source[0] = source[0]+256
        source[1] = source[1]+192
        _, px_target, inliers = projection.warp(source,
                                                source_position_map,
                                                target_R_CW, target_t_CW,
                                                mask_fov=True,
                                                mask_occlusion=target_position_map,
                                                mask_reflectance=source_reflectance_map)
        px_target[0] = px_target[0]-256
        px_target[1] = px_target[1]-192
        px_target[0] = px_target[0] - this_scenter2tcenter[0]
        px_target[1] = px_target[1] - this_scenter2tcenter[1]
        # wrapped again
        px_target = px_target.view(C, H, W)
        inliers = inliers.view(1, H, W)
        target_name = str(target_frame[0][b].item()) \
                    + '_' + str(target_frame[1][b].item()) \
                    + '_' + str(target_frame[2][b].item()) \
                    + '_' + str(target_frame[3][b].item())
        warped_sources.append(px_target)
        inliers_list.append(inliers)
        source_name_list.append(source_name)
        target_name_list.append(target_name)
    return torch.stack(warped_sources, dim=0), torch.stack(inliers_list, dim=0), source_name_list, target_name_list


class KeypointNetwithIOLoss(torch.nn.Module):
    """
    Model class encapsulating the KeypointNet and the IONet.

    Parameters
    ----------
    keypoint_loss_weight: float
        Keypoint loss weight.
    descriptor_loss_weight: float
        Descriptor loss weight.
    score_loss_weight: float
        Score loss weight.
    keypoint_net_learning_rate: float
        Keypoint net learning rate.
    with_io:
        Use IONet.
    use_color : bool
        Use color or grayscale images.
    do_upsample: bool
        Upsample desnse descriptor map.
    do_cross: bool
        Predict keypoints outside cell borders.
    with_drop : bool
        Use dropout.
    descriptor_loss: bool
        Use descriptor loss.
    kwargs : dict
        Extra parameters
    """
    def __init__(
        self, training_mode, keypoint_loss_weight=1.0, descriptor_loss_weight=2.0, score_loss_weight=1.0, 
        keypoint_net_learning_rate=0.001, with_io=True, use_color=True, do_upsample=True, 
        do_cross=True, descriptor_loss=True, with_drop=True, keypoint_net_type='KeypointNet',
        pretrained_model=None, **kwargs):

        super().__init__()

        self.keypoint_loss_weight = keypoint_loss_weight
        self.descriptor_loss_weight = descriptor_loss_weight
        self.score_loss_weight = score_loss_weight
        self.keypoint_net_learning_rate = keypoint_net_learning_rate
        self.optim_params = []

        self.cell = 8 # Size of each output cell. Keep this fixed.
        self.border_remove = 4 # Remove points this close to the border.
        self.top_k2 = 300
        self.relax_field = 4

        self.use_color = use_color
        self.descriptor_loss = descriptor_loss
        self.training_mode=training_mode

        # Initialize KeypointNet
        if pretrained_model == None:
            if keypoint_net_type == 'KeypointNet':
                self.keypoint_net = KeypointNet(use_color=use_color, do_upsample=do_upsample, with_drop=with_drop, do_cross=do_cross)
            elif keypoint_net_type == 'KeypointResnet':
                self.keypoint_net = KeypointResnet(with_drop=with_drop)
            else:
                raise NotImplemented('Keypoint net type not supported {}'.format(keypoint_net_type))
        else:
            checkpoint = torch.load(pretrained_model)
            model_args = checkpoint['config']['model']['params']
            if 'keypoint_net_type' in checkpoint['config']['model']['params']:
                net_type = checkpoint['config']['model']['params']
            else:
                net_type = KeypointNet # default when no type is specified
            if net_type is KeypointNet:
                print('keypointNet')
                self.keypoint_net = KeypointNet(use_color=model_args['use_color'],
                                do_upsample=model_args['do_upsample'],
                                do_cross=model_args['do_cross'])
            else:
                print('keypointresnet')
                self.keypoint_net = KeypointResnet()
            self.keypoint_net.load_state_dict(checkpoint['state_dict'])
            print('Loaded KeypointNet from {}'.format(pretrained_model))
            print('KeypointNet params {}'.format(model_args))

        self.keypoint_net = self.keypoint_net.cuda()
        self.add_optimizer_params('KeypointNet', self.keypoint_net.parameters(), keypoint_net_learning_rate)

        self.with_io = with_io
        self.io_net = None
        if self.with_io:
            self.io_net = InlierNet(blocks=4)
            self.io_net = self.io_net.cuda()
            self.add_optimizer_params('InlierNet', self.io_net.parameters(),  keypoint_net_learning_rate)

        self.train_metrics = {}
        self.vis = {}
        if torch.cuda.current_device() == 0:
            print('KeypointNetwithIOLoss:: with io {} with descriptor loss {}'.format(self.with_io, self.descriptor_loss))

    def add_optimizer_params(self, name, params, lr):
        self.optim_params.append(
            {'name': name, 'lr': lr, 'original_lr': lr,
             'params': filter(lambda p: p.requires_grad, params)})

    def forward(self, data, debug=False):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch.
        debug : bool
            True if to compute debug data (stored in self.vis).

        Returns
        -------
        output : dict
            Dictionary containing the output of depth and pose networks
        """

        loss_2d = 0

        if self.training:

            B, _, H, W = data['image'].shape
            device = data['image'].device

            reprojection = Reprojection(width=1024, height=768, verbose=False)

            recall_2d = 0
            inlier_cnt = 0

            input_img_0 = data['image']
            input_img_aug_0 = data['image_aug']
            # metainfo: [source_position_map, target_position_map, source_reflectance_map, target_R_CW, target_t_CW]
            metainfo = data['metainfo']
            source_frame = data['source_frame']
            target_frame = data['target_frame']
            scenter2tcenter = data['scenter2tcenter']

            input_img = to_color_normalized(input_img_0.clone())
            input_img_aug = to_color_normalized(input_img_aug_0.clone())

            # Get network outputs
            # score: (B, 1, H_out, W_out)
            # uv_pred: (B, 2, H_out, W_out)
            # feat: (B, 256, H_out, W_out)
            source_score, source_uv_pred, source_feat = self.keypoint_net(input_img_aug)
            target_score, target_uv_pred, target_feat = self.keypoint_net(input_img)
            _, _, Hc, Wc = target_score.shape

            # Normalize uv coordinates
            # TODO: Have a function for the norm and de-norm of 2d coordinate.
            target_uv_norm = target_uv_pred.clone()
            target_uv_norm[:,0] = (target_uv_norm[:,0] / (float(W-1)/2.)) - 1.
            target_uv_norm[:,1] = (target_uv_norm[:,1] / (float(H-1)/2.)) - 1.
            target_uv_norm = target_uv_norm.permute(0, 2, 3, 1)

            source_uv_norm = source_uv_pred.clone()
            source_uv_norm[:,0] = (source_uv_norm[:,0] / (float(W-1)/2.)) - 1.
            source_uv_norm[:,1] = (source_uv_norm[:,1] / (float(H-1)/2.)) - 1.
            source_uv_norm = source_uv_norm.permute(0, 2, 3, 1)

            if self.training_mode=='scene' or self.training_mode=='cam' or self.training_mode=='con'\
                or self.training_mode=='scene+HA' or self.training_mode=='cam+HA' or self.training_mode=='con+HA':
                # get source_uv with frame transformation, then normalize
                # source_uv_pred dim: (B, 2, H_out, W_out)
                source_uv_warped, inliers, _, _ = warp_frame2frame_batch(source_uv_pred, metainfo, source_frame, target_frame, scenter2tcenter, projection=reprojection)
                source_uv_warped = source_uv_warped.float()
                # normalization
                source_uv_warped_norm = source_uv_warped.clone()
                source_uv_warped_norm[:,0] = (source_uv_warped_norm[:,0] / (float(W-1)/2.)) - 1.
                source_uv_warped_norm[:,1] = (source_uv_warped_norm[:,1] / (float(H-1)/2.)) - 1.
                source_uv_warped_norm = source_uv_warped_norm.permute(0, 2, 3, 1)
            if self.training_mode=='HA' or self.training_mode=='HA_wo_sp':
                homography = data['homography']
                source_uv_warped_norm = warp_homography_batch(source_uv_norm, homography)
                source_uv_warped = source_uv_warped_norm.clone()
                source_uv_warped[:,:,:,0] = (source_uv_warped[:,:,:,0] + 1) * (float(W-1)/2.)
                source_uv_warped[:,:,:,1] = (source_uv_warped[:,:,:,1] + 1) * (float(H-1)/2.)    # (B,H,W,C)
                source_uv_warped = source_uv_warped.permute(0, 3, 1, 2)    # (B,C,H,W)
            if self.training_mode=='scene+HA' or self.training_mode=='cam+HA' or self.training_mode=='con+HA':
                homography = data['homography']
                source_uv_warped_norm = warp_homography_batch(source_uv_warped_norm, homography)
                source_uv_warped = source_uv_warped_norm.clone()
                source_uv_warped[:,:,:,0] = (source_uv_warped[:,:,:,0] + 1) * (float(W-1)/2.)
                source_uv_warped[:,:,:,1] = (source_uv_warped[:,:,:,1] + 1) * (float(H-1)/2.)    # (B,H,W,C)
                source_uv_warped = source_uv_warped.permute(0, 3, 1, 2)    # (B,C,H,W)

            target_uv_resampled = torch.nn.functional.grid_sample(target_uv_pred, source_uv_warped_norm.float(), mode='nearest', align_corners=True)

            target_uv_resampled_norm = target_uv_resampled.clone()
            target_uv_resampled_norm[:,0] = (target_uv_resampled_norm[:,0] / (float(W-1)/2.)) - 1.
            target_uv_resampled_norm[:,1] = (target_uv_resampled_norm[:,1] / (float(H-1)/2.)) - 1.
            target_uv_resampled_norm = target_uv_resampled_norm.permute(0, 2, 3, 1)

            # Border mask
            border_mask_ori = torch.ones(B,Hc,Wc)
            border_mask_ori[:,0] = 0
            border_mask_ori[:,Hc-1] = 0
            border_mask_ori[:,:,0] = 0
            border_mask_ori[:,:,Wc-1] = 0
            border_mask_ori = border_mask_ori.gt(1e-3).to(device)

            # Out-of-bourder(OOB) mask. Not nessesary in our case, since it's prevented at HA procedure already. Kept here for future usage.
            oob_mask2 = source_uv_warped_norm[:,:,:,0].lt(1) & source_uv_warped_norm[:,:,:,0].gt(-1) & source_uv_warped_norm[:,:,:,1].lt(1) & source_uv_warped_norm[:,:,:,1].gt(-1)
            border_mask = border_mask_ori & oob_mask2
            if not (self.training_mode=='HA' or self.training_mode=='HA_wo_sp'):
                # Treat outliers from Hypersim projection as out of bournder points.
                inliers = inliers.squeeze()
                border_mask = border_mask & inliers

            d_uv_mat_abs = torch.abs(source_uv_warped.view(B,2,-1).unsqueeze(3) - target_uv_pred.view(B,2,-1).unsqueeze(2))
            d_uv_l2_mat = torch.norm(d_uv_mat_abs, p=2, dim=1)
            d_uv_l2_min, d_uv_l2_min_index = d_uv_l2_mat.min(dim=2)

            dist_norm_valid_mask = d_uv_l2_min.lt(4) & border_mask.view(B,Hc*Wc)

            # Keypoint loss
            loc_loss = d_uv_l2_min[dist_norm_valid_mask].mean()
            loss_2d += self.keypoint_loss_weight * loc_loss.mean()

            # Desc Head Loss, per-pixel level triplet loss from https://arxiv.org/pdf/1902.11046.pdf.
            if self.descriptor_loss:
                metric_loss, recall_2d = build_descriptor_loss(source_feat, target_feat, source_uv_norm.detach(), source_uv_warped_norm.detach(), source_uv_warped, keypoint_mask=border_mask, relax_field=self.relax_field)
                loss_2d += self.descriptor_loss_weight * metric_loss * 2
            else:
                _, recall_2d = build_descriptor_loss(source_feat, target_feat, source_uv_norm, source_uv_warped_norm, source_uv_warped, keypoint_mask=border_mask, relax_field=self.relax_field, eval_only=True)

            #Score Head Loss
            target_score_associated = target_score.view(B,Hc*Wc).gather(1, d_uv_l2_min_index).view(B,Hc,Wc).unsqueeze(1)
            dist_norm_valid_mask = dist_norm_valid_mask.view(B,Hc,Wc).unsqueeze(1) & border_mask.unsqueeze(1)
            d_uv_l2_min = d_uv_l2_min.view(B,Hc,Wc).unsqueeze(1)
            loc_err = d_uv_l2_min[dist_norm_valid_mask]

            usp_loss = (target_score_associated[dist_norm_valid_mask] + source_score[dist_norm_valid_mask]) * (loc_err - loc_err.mean())
            loss_2d += self.score_loss_weight * usp_loss.mean()

            target_score_resampled = torch.nn.functional.grid_sample(target_score, source_uv_warped_norm.detach(), mode='bilinear', align_corners=True)

            loss_2d += self.score_loss_weight * torch.nn.functional.mse_loss(target_score_resampled[border_mask.unsqueeze(1)],
                                                                                source_score[border_mask.unsqueeze(1)]).mean() * 2
            if self.with_io:
                # Compute IO loss
                top_k_score1, top_k_indice1 = source_score.view(B,Hc*Wc).topk(self.top_k2, dim=1, largest=False)
                top_k_mask1 = torch.zeros(B, Hc * Wc).to(device)
                top_k_mask1.scatter_(1, top_k_indice1, value=1)
                top_k_mask1 = top_k_mask1.gt(1e-3).view(B,Hc,Wc)

                top_k_score2, top_k_indice2 = target_score.view(B,Hc*Wc).topk(self.top_k2, dim=1, largest=False)
                top_k_mask2 = torch.zeros(B, Hc * Wc).to(device)
                top_k_mask2.scatter_(1, top_k_indice2, value=1)
                top_k_mask2 = top_k_mask2.gt(1e-3).view(B,Hc,Wc)

                source_uv_norm_topk = source_uv_norm[top_k_mask1].view(B, self.top_k2, 2)
                target_uv_norm_topk = target_uv_norm[top_k_mask2].view(B, self.top_k2, 2)
                source_uv_warped_norm_topk = source_uv_warped_norm[top_k_mask1].view(B, self.top_k2, 2)

                source_feat_topk = torch.nn.functional.grid_sample(source_feat, source_uv_norm_topk.unsqueeze(1), align_corners=True).squeeze()
                target_feat_topk = torch.nn.functional.grid_sample(target_feat, target_uv_norm_topk.unsqueeze(1), align_corners=True).squeeze()

                source_feat_topk = source_feat_topk.div(torch.norm(source_feat_topk, p=2, dim=1).unsqueeze(1))
                target_feat_topk = target_feat_topk.div(torch.norm(target_feat_topk, p=2, dim=1).unsqueeze(1))

                dmat = torch.bmm(source_feat_topk.permute(0,2,1), target_feat_topk)
                dmat = torch.sqrt(2 - 2 * torch.clamp(dmat, min=-1, max=1))
                dmat_soft_min = torch.sum(dmat* dmat.mul(-1).softmax(dim=2), dim=2)
                dmat_min, dmat_min_indice = torch.min(dmat, dim=2)

                target_uv_norm_topk_associated = target_uv_norm_topk.gather(1, dmat_min_indice.unsqueeze(2).repeat(1,1,2))
                point_pair = torch.cat([source_uv_norm_topk, target_uv_norm_topk_associated, dmat_min.unsqueeze(2)], 2)

                inlier_pred = self.io_net(point_pair.permute(0,2,1).unsqueeze(3)).squeeze()

                target_uv_norm_topk_associated_raw = target_uv_norm_topk_associated.clone()
                target_uv_norm_topk_associated_raw[:,:,0] = (target_uv_norm_topk_associated_raw[:,:,0] + 1) * (float(W-1)/2.)
                target_uv_norm_topk_associated_raw[:,:,1] = (target_uv_norm_topk_associated_raw[:,:,1] + 1) * (float(H-1)/2.)

                source_uv_warped_norm_topk_raw = source_uv_warped_norm_topk.clone()
                source_uv_warped_norm_topk_raw[:,:,0] = (source_uv_warped_norm_topk_raw[:,:,0] + 1) * (float(W-1)/2.)
                source_uv_warped_norm_topk_raw[:,:,1] = (source_uv_warped_norm_topk_raw[:,:,1] + 1) * (float(H-1)/2.)


                matching_score = torch.norm(target_uv_norm_topk_associated_raw - source_uv_warped_norm_topk_raw, p=2, dim=2)
                inlier_mask = matching_score.lt(4)
                inlier_gt = 2 * inlier_mask.float() - 1

                if inlier_mask.sum() > 10:

                    io_loss = torch.nn.functional.mse_loss(inlier_pred, inlier_gt)
                    loss_2d += self.keypoint_loss_weight * io_loss


            if debug and torch.cuda.current_device() == 0:
                # Generate visualization data
                vis_ori = (input_img[0].permute(1, 2, 0).detach().cpu().clone().squeeze() )
                vis_ori -= vis_ori.min()
                vis_ori /= vis_ori.max()
                vis_ori = (vis_ori* 255).numpy().astype(np.uint8)

                vis_tar = (input_img[0].permute(1, 2, 0).detach().cpu().clone().squeeze() )
                vis_tar -= vis_tar.min()
                vis_tar /= vis_tar.max()
                vis_tar = (vis_tar* 255).numpy().astype(np.uint8)

                vis_src = (input_img_aug[0].permute(1, 2, 0).detach().cpu().clone().squeeze() )
                vis_src -= vis_src.min()
                vis_src /= vis_src.max()
                vis_src = (vis_src* 255).numpy().astype(np.uint8)

                if self.use_color is False:
                    vis_ori = cv2.cvtColor(vis_ori, cv2.COLOR_GRAY2BGR)

                _, top_k = target_score.view(B,-1).topk(self.top_k2, dim=1) #JT: Target frame keypoints
                vis_ori = draw_keypoints(vis_ori, target_uv_pred.view(B,2,-1)[:,:,top_k[0].squeeze()],(0,0,255))

                _, top_k = source_score.view(B,-1).topk(self.top_k2, dim=1) #JT: Warped Source frame keypoints
                vis_ori = draw_keypoints(vis_ori, source_uv_warped.view(B,2,-1)[:,:,top_k[0].squeeze()],(255,0,255))

                cm = get_cmap('plasma')
                heatmap = target_score[0].detach().cpu().clone().numpy().squeeze()
                heatmap -= heatmap.min()
                heatmap /= heatmap.max()
                heatmap = cv2.resize(heatmap, (W, H))
                heatmap = cm(heatmap)[:, :, :3]

                self.vis['img_ori'] = np.clip(vis_ori, 0, 255) / 255.
                self.vis['heatmap'] = np.clip(heatmap * 255, 0, 255) / 255.
                # Visualization of projection. Uncomment to activate---does not work if use frame2frame combined with HA
                # self.vis['img_src'] = np.clip(vis_src, 0, 255) / 255.
                # self.vis['img_tar'] = np.clip(vis_tar, 0, 255) / 255.

                # import itertools
                # W_a = [*range(0, 512, 4)]
                # H_a = [*range(0, 384, 4)]
                # px_source = torch.tensor(list(itertools.product(W_a, H_a))).T.float().cuda()
                # px_source = px_source.view(2, len(H_a), len(W_a))
                # px_source = torch.unsqueeze(px_source, 0)
                # # source_uv_warped, inliers = warp_frame2frame_batch(
                # #     px_source, torch.unsqueeze(metainfo[0], 0), torch.unsqueeze(source_frame[0], 0), 
                # #     torch.unsqueeze(target_frame[0], 0), torch.unsqueeze(scenter2tcenter[0], 0), projection=reprojection)
                # source_uv_warped, inliers, source_name, target_name = warp_frame2frame_batch(
                #     px_source, metainfo, source_frame, 
                #     target_frame, scenter2tcenter, projection=reprojection)
                # source_uv_warped = source_uv_warped
                # inliers = inliers.squeeze()
                # outliers = ~inliers
                # source_uv_inliers = px_source[:,:,inliers]
                # source_uv_outliers = px_source[:,:,outliers]
                # source_uv_warped_inliers = source_uv_warped[:,:,inliers]
                # source_uv_warped_outliers = source_uv_warped[:,:,outliers]

                # vis_src_masked = draw_keypoints(vis_src, source_uv_outliers,(255,0,255))
                # vis_src_masked = draw_keypoints(vis_src_masked, source_uv_inliers,(0,0,255))
                
                # vis_tar_masked = draw_keypoints(vis_tar, source_uv_warped_outliers,(255,0,255))
                # vis_tar_masked = draw_keypoints(vis_tar_masked, source_uv_warped_inliers,(0,0,255))
                
                # self.vis['img_src_masked'] = np.clip(vis_src_masked, 0, 255) / 255.
                # self.vis['img_tar_masked'] = np.clip(vis_tar_masked, 0, 255) / 255.
                # self.vis['source_name'] = source_name[0]
                # self.vis['target_name'] = target_name[0]

        return loss_2d, recall_2d
