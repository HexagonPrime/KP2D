import numpy as np
import torch

class Reprojection:
    def __init__(self, width=1024, height=768, verbose=False, device='gpu'):
        self.verbose = verbose
        self.H = height
        self.W = width
        self.K = self.computeProjectionMatrix()
        self.device = device

    # Given a pixel in source image coordinates, project it into image coordinates of the target frame.
    # Input: Pixel in source coordinates (width, height), pixel position map source frame, camera pose target frame
    def projectPixelFromSourceToTarget(self, px_source, source_position_map, R_CW, C_t_CW):
        # Get world position of source pixel.
        px_source_int = np.rint(px_source).astype(int)
        W_t_WP = source_position_map[px_source_int[1], px_source_int[0]]
        if self.verbose:
            print('The source pixel [{:d}, {:d}] corresponds to the point [{:.2f}, {:.2f}, {:.2f}] in world coordinates.'.format(px_source_int[0], px_source_int[1], W_t_WP[0], W_t_WP[1], W_t_WP[2]))

        # Transform into target camera using the target cam
        return self.transformPointWorldToScreen(R_CW, C_t_CW, W_t_WP)

    def projectPixelFromSourceToTargetInt(self, px_source, source_position_map, R_CW, C_t_CW):
        px_target = self.projectPixelFromSourceToTarget(px_source, source_position_map, R_CW, C_t_CW)
        return np.rint(px_target).astype(int)

    # Get the projection matrix that projects a homogeneous point in world coordinates onto screen coordinates.
    def getProjectionMatrix(self, R_CW, C_t_CW):
        C_T_CW = np.block([[R_CW, np.asmatrix(C_t_CW).T], [np.zeros((1,3)), np.ones((1,1))]])
        return self.K.dot(C_T_CW)

    # Warp source pixels to target pixels using torch.
    # Input: source pixel tensor, first column: width, second column height.
    #        map from pixels to world positions of source frame
    #        rotation of target frame camera
    #        translation of target frame camera
    #        (optional) boolean to mask pixel that are out of field of view of target cam
    #        (optional) map from pixels to world positions of target frame to mask occluded pixel
    #        (optional) occlusion threshold in meters
    #        (optional) map from pixels to reflectance values of source frame to mask reflectant pixel
    #        (optional) reflectance threshold
    # Output: All source pixels, all projected target pixels, list of inliers.
    # Example:
    # px_src, px_trg, inliers = warp(...)
    # px_src_inliers = px_src[:,inliers]
    # px_trg_inliers = px_trg[:,inliers]
    def warp(self, px_source, source_position_map, R_CW, C_t_CW_0, mask_fov=False, mask_occlusion=None, occlusion_threshold=0.03, mask_reflectance=None, reflectance_threshold=30):
        if self.device=='gpu':
            source_position_map=source_position_map.cuda()
        K = torch.from_numpy(self.K).float()
        if self.device=='gpu':
            K = K.cuda()
        N = px_source.size(1)

        # Construct projection matrix.
        C_T_CW = torch.eye(4)
        if self.device == 'gpu':
            C_T_CW = C_T_CW.cuda()
        C_T_CW[0:3, 0:3] = R_CW
        C_T_CW[0:3, 3] = C_t_CW_0
        if self.device=='gpu':
            C_T_CW = C_T_CW.cuda()
        P = torch.matmul(K, C_T_CW)

        # Get homogeneous world position of pixels.
        if self.device == 'gpu':
                W_t_WP_0 = source_position_map[torch.round(px_source[1,:]).long(), torch.round(px_source[0,:]).long()].T
        else:
            W_t_WP_0 = source_position_map[px_source[1,:], px_source[0,:]].T

        if self.device=='gpu':
            try:
                W_t_WP = torch.cat((W_t_WP_0, torch.ones(1, N).cuda()), 0)
            except RuntimeError:
                torch.save(px_source, 'px_source.pt')
                torch.save(W_t_WP_0, 'W_t_WP_0.pt')
        else:
            W_t_WP = torch.cat((W_t_WP_0, torch.ones(1, N)), 0)

        # Project pixels into screen coordinates.
        p_screen = torch.matmul(P, W_t_WP)

        # Normalize.
        p_screen = p_screen / p_screen[3,:]

        # Compute pixel coordinates from relative points around camera center.
        px_target = torch.zeros(2, N)
        if self.device=='gpu':
            px_target = px_target.cuda()
        px_target[0,:] = 0.5 * (p_screen[0,:] + 1) * (self.W - 1)
        px_target[1,:] = (1 - 0.5 * (p_screen[1,:] + 1)) * (self.H - 1)

        # Round to integer
        px_target = px_target.round().long()

        # Masking
        # WARNING: The order of the checks matters!
        inlier_mask = torch.arange(0, N)
        if self.device=='gpu':
            inlier_mask = inlier_mask.cuda()

        # Mask pixel that are outside of field of view.
        if mask_fov or mask_occlusion is not None:
            if self.device=='gpu':
                inlier_mask = torch.logical_and(inlier_mask, (px_target[0,:]>=0).cuda())
                inlier_mask = torch.logical_and(inlier_mask, (px_target[1,:]>=0).cuda())
                inlier_mask = torch.logical_and(inlier_mask, (px_target[0,:]<self.W).cuda())
                inlier_mask = torch.logical_and(inlier_mask, (px_target[1,:]<self.H).cuda())
            else:
                inlier_mask = torch.logical_and(inlier_mask, px_target[0,:]>=0)
                inlier_mask = torch.logical_and(inlier_mask, px_target[1,:]>=0)
                inlier_mask = torch.logical_and(inlier_mask, px_target[0,:]<self.W)
                inlier_mask = torch.logical_and(inlier_mask, px_target[1,:]<self.H)


        # Mask pixel that are occluded.
        if mask_occlusion is not None:
            # Get world position of target pixels that are in FOV.
            # target_position_map = torch.from_numpy(mask_occlusion)
            target_position_map = mask_occlusion
            if self.device=='gpu':
                target_position_map = target_position_map.cuda()
            W_t_WP_source = source_position_map[torch.round(px_source[1,inlier_mask]).long(), torch.round(px_source[0,inlier_mask]).long()]
            W_t_WP_target = target_position_map[px_target[1,inlier_mask], px_target[0,inlier_mask]]
            inlier_mask[inlier_mask==True] = torch.logical_and(inlier_mask[inlier_mask==True], torch.norm(W_t_WP_target - W_t_WP_source, dim=1) < occlusion_threshold)

        # Mask pixel that are reflectant.
        if mask_reflectance is not None:
            # source_reflectance_map = torch.from_numpy(mask_reflectance)
            source_reflectance_map = mask_reflectance
            if self.device=='gpu':
                source_reflectance_map = source_reflectance_map.cuda()
            reflectance = source_reflectance_map[torch.round(px_source[1,inlier_mask]).long(), torch.round(px_source[0,inlier_mask]).long()]
            inlier_mask[inlier_mask==True] = torch.logical_and(inlier_mask[inlier_mask==True], torch.any(reflectance >= reflectance_threshold, dim=1))

        return px_source, px_target, inlier_mask


    # Transform a 3D point from world coordinates into camera coordinates given the camera pose.
    # Returns [px_width, px_height, depth]
    def transformPointWorldToScreen(self, R_CW, C_t_CW, W_t_WP):
        # Project point onto screen.
        P = self.getProjectionMatrix(R_CW, C_t_CW)
        W_t_WP_hom = np.concatenate((W_t_WP, [1]), axis=None)
        p_screen = np.squeeze(np.array(P.dot(W_t_WP_hom)))

        # Normalize
        p_screen = p_screen / p_screen[3]
        # Compute pixel coordinates from relative points around camera center.
        p_uvd = np.zeros(3)
        p_uvd[0] = 0.5 * (p_screen[0] + 1) * (self.W - 1)
        p_uvd[1] = (1 - 0.5 * (p_screen[1] + 1)) * (self.H - 1)
        p_uvd[2] = (p_screen[2] + 1) / 2.0
        if self.verbose:
            print('W_t_WP [{:.2f}, {:.2f}, {:.2f}] converted into image coordinates [{:.2f}, {:.2f}, {:.2f}]'.format(W_t_WP[0], W_t_WP[1], W_t_WP[2], p_uvd[0], p_uvd[1], p_uvd[2]))
        return p_uvd

    def transformPointWorldToScreenInt(self, R_CW, C_t_CW, W_t_WP):
        p_uvd = self.transformPointWorldToScreen(R_CW, C_t_CW, W_t_WP)
        return np.rint(p_uvd).astype(int)

    def isFieldOfView(self, p_screen):
        return np.all(p_screen >= 0) and p_screen[0] < self.W and p_screen[1] < self.H

    # Return camera matrix K in 4x4 to project 3D coordinates in camera frame to image plane.
    # https://github.com/apple/ml-hypersim/blob/7bc2a8a751c0157c1bd956972acbdd2ddf85186c/code/python/tools/scene_generate_images_bounding_box.py#L129-L149
    def computeProjectionMatrix(self):
        fov_x = np.pi / 3.0
        fov_y = 2.0 * np.arctan(self.H * np.tan(fov_x / 2.0) / self.W)
        near = 1.0
        far = 1000.0

        f_h = np.tan(fov_y / 2.0) * near
        f_w = f_h * self.W / self.H
        left = -f_w
        right = f_w
        bottom = -f_h
        top = f_h

        K = np.matrix(np.zeros((4, 4)))
        K[0, 0] = (2.0 * near) / (right - left)
        K[1, 1] = (2.0 * near) / (top - bottom)
        K[0, 2] = (right + left) / (right - left)
        K[1, 2] = (top + bottom) / (top - bottom)
        K[2, 2] = -(far + near) / (far - near)
        K[3, 2] = -1.0
        K[2, 3] = -(2.0 * far * near) / (far - near)
        if self.verbose:
            print('Camera projection matrix:')
            print(K)
        return K