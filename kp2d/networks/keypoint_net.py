# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
from math import pi

import torch
import torch.nn.functional as F
from PIL import Image

from kp2d.utils.image import image_grid


class KeypointNet(torch.nn.Module):
    """
    Keypoint detection network.

    Parameters
    ----------
    use_color : bool
        Use color or grayscale images.
    do_upsample: bool
        Upsample desnse descriptor map.
    with_drop : bool
        Use dropout.
    do_cross: bool
        Predict keypoints outside cell borders.
    kwargs : dict
        Extra parameters
    """

    def __init__(self, use_color=True, do_upsample=True, with_drop=True, do_cross=True, **kwargs):
        super().__init__()

        self.training = True

        self.use_color = use_color
        self.with_drop = with_drop
        self.do_cross = do_cross
        self.do_upsample = do_upsample

        if self.use_color:
            c0 = 3
        else:
            c0 = 1

        self.bn_momentum = 0.1
        self.cross_ratio = 2.0

        if self.do_cross is False:
            self.cross_ratio = 1.0

        c1, c2, c3, c4, c5, d1 = 32, 64, 128, 256, 256, 512

        self.conv1a = torch.nn.Sequential(torch.nn.Conv2d(c0, c1, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c1,momentum=self.bn_momentum))
        self.conv1b = torch.nn.Sequential(torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c1,momentum=self.bn_momentum))
        self.conv2a = torch.nn.Sequential(torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c2,momentum=self.bn_momentum))
        self.conv2b = torch.nn.Sequential(torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c2,momentum=self.bn_momentum))
        self.conv3a = torch.nn.Sequential(torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c3,momentum=self.bn_momentum))
        self.conv3b = torch.nn.Sequential(torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c3,momentum=self.bn_momentum))
        self.conv4a = torch.nn.Sequential(torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        self.conv4b = torch.nn.Sequential(torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))

        # Score Head.
        self.convDa = torch.nn.Sequential(torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        self.convDb = torch.nn.Conv2d(c5, 1, kernel_size=3, stride=1, padding=1)

        # Location Head.
        self.convPa = torch.nn.Sequential(torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        self.convPb = torch.nn.Conv2d(c5, 2, kernel_size=3, stride=1, padding=1)

        # Desc Head.
        self.convFa = torch.nn.Sequential(torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        self.convFb = torch.nn.Sequential(torch.nn.Conv2d(c5, d1, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(d1,momentum=self.bn_momentum))
        self.convFaa = torch.nn.Sequential(torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c5,momentum=self.bn_momentum))
        self.convFbb = torch.nn.Conv2d(c5, 256, kernel_size=3, stride=1, padding=1)

        self.relu = torch.nn.LeakyReLU(inplace=True)
        if self.with_drop:
            self.dropout = torch.nn.Dropout2d(0.2)
        else:
            self.dropout = None
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.cell = 8
        self.upsample = torch.nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        """
        Processes a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images (B, 3, H, W)

        Returns
        -------
        score : torch.Tensor
            Score map (B, 1, H_out, W_out)
        coord: torch.Tensor
            Keypoint coordinates (B, 2, H_out, W_out)
        feat: torch.Tensor
            Keypoint descriptors (B, 256, H_out, W_out)
        """
        B, _, H, W = x.shape

        if torch.any(x.isnan()):
            print('print -1')
            x = torch.nan_to_num(x, nan=0, posinf=0)

        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        skip = self.relu(self.conv3b(x))
        if self.dropout:
            skip = self.dropout(skip)
        x = self.pool(skip)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        if self.dropout:
            x = self.dropout(x)
        if torch.any(x.isnan()):
            print('print 0')

        B, _, Hc, Wc = x.shape

        score = self.relu(self.convDa(x))
        if torch.any(score.isnan()):
            print('print 1')
        if self.dropout:
            score = self.dropout(score)
        if torch.any(score.isnan()):
            print('print 2')
        score = self.convDb(score).sigmoid()
        if torch.any(score.isnan()):
            print('print 3')

        border_mask = torch.ones(B, Hc, Wc)
        border_mask[:, 0] = 0
        border_mask[:, Hc - 1] = 0
        border_mask[:, :, 0] = 0
        border_mask[:, :, Wc - 1] = 0
        border_mask = border_mask.unsqueeze(1)
        if torch.any(border_mask.isnan()):
            print('print 4')
        score = score * border_mask.to(score.device)
        if torch.any(score.isnan()):
            print('print 5')

        center_shift = self.relu(self.convPa(x))
        if torch.any(center_shift.isnan()):
            print('print 6')
        if self.dropout:
            center_shift = self.dropout(center_shift)
        if torch.any(center_shift.isnan()):
            print('print 7')
        center_shift = self.convPb(center_shift).tanh()
        if torch.any(center_shift.isnan()):
            print('print 8')

        step = (self.cell-1) / 2.
        center_base = image_grid(B, Hc, Wc,
                                 dtype=center_shift.dtype,
                                 device=center_shift.device,
                                 ones=False, normalized=False).mul(self.cell) + step
        if torch.any(center_base.isnan()):
            print('print 9')

        coord_un = center_base.add(center_shift.mul(self.cross_ratio * step))
        if torch.any(coord_un.isnan()):
            print('print 10')
        coord = coord_un.clone()
        coord[:, 0] = torch.clamp(coord_un[:, 0], min=0, max=W-1)
        if torch.any(coord.isnan()):
            print('print 11')
        coord[:, 1] = torch.clamp(coord_un[:, 1], min=0, max=H-1)
        if torch.any(coord.isnan()):
            print('print 12')

        feat = self.relu(self.convFa(x))
        if torch.any(feat.isnan()):
            print('print 13')
        if self.dropout:
            feat = self.dropout(feat)
        if torch.any(feat.isnan()):
            print('print 14')
        if self.do_upsample:
            feat = self.upsample(self.convFb(feat))
            if torch.any(feat.isnan()):
                print('print 15')
            feat = torch.cat([feat, skip], dim=1)
            if torch.any(feat.isnan()):
                print('print 16')
        feat = self.relu(self.convFaa(feat))
        if torch.any(feat.isnan()):
            print('print 17')
        feat = self.convFbb(feat)
        if torch.any(feat.isnan()):
            print('print 18')

        if self.training is False:
            coord_norm = coord[:, :2].clone()
            if torch.any(coord_norm.isnan()):
                print('print 19')
            coord_norm[:, 0] = (coord_norm[:, 0] / (float(W-1)/2.)) - 1.
            if torch.any(coord_norm.isnan()):
                print('print 20')
            coord_norm[:, 1] = (coord_norm[:, 1] / (float(H-1)/2.)) - 1.
            if torch.any(coord_norm.isnan()):
                print('print 21')
            coord_norm = coord_norm.permute(0, 2, 3, 1)
            if torch.any(coord_norm.isnan()):
                print('print 22')

            feat = torch.nn.functional.grid_sample(feat, coord_norm, align_corners=True)
            if torch.any(feat.isnan()):
                print('print 23')

            dn = torch.norm(feat, p=2, dim=1)  # Compute the norm.
            if torch.any(dn.isnan()):
                print('print 24')
            feat = feat.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
            if torch.any(feat.isnan()):
                print('print 25')
        return score, coord, feat
