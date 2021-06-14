import glob
import os

from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import h5py
import numpy as np
from kp2d.datasets.data_loader import DataLoader
from torchvision.transforms.functional import crop
import torchvision.transforms as transforms
from itertools import combinations
import torch
from kp2d.utils.reprojection import Reprojection

class HypersimLoader(Dataset):
    """
    Hypersim dataset class.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    data_transform : Function
        Transformations applied to the sample
    """
    def __init__(self, root_dir, training_mode, interval, data_transform=None, center_crop=True, partition='train'):

        super().__init__()
        self.loading_mode = 'jpg'
        self.data_util_loader = DataLoader(root_dir, image_type=self.loading_mode, verbose=False)
        self.reprojection = Reprojection(width=1024, height=768, device='cpu', verbose=False)
        self.center_crop = center_crop

        print('Training mode: ' + training_mode)
        self.root_dir = root_dir

        col_list = ["volume", "scene", "cam", "frame"]
        # df = pd.read_csv(os.path.join(root_dir, 'hypersim_0.csv'), names=col_list)
        if partition=='val+test':
            df = [pd.read_csv(os.path.join(root_dir, 'true_frames_2_0.2_val.csv'), names=col_list),
                    pd.read_csv(os.path.join(root_dir, 'true_frames_2_0.2_test.csv'), names=col_list)]
            df = pd.concat(df, ignore_index=True)
        elif partition=='weakly_filtered':
            df = pd.read_csv(os.path.join(root_dir, 'true_frames.csv'), names=col_list)
        elif partition=='strongly_filtered':
            df = pd.read_csv(os.path.join(root_dir, 'true_frames_8_2.0_train.csv'), names=col_list)
        else:
            raise RuntimeError('Hypersim partition invalid!')
        self.data_transform = data_transform

        # mode can be: 'HA', 'scene' or 'cam'
        self.training_mode = training_mode
        self.files = []
        if training_mode == "scene" or training_mode=='scene+HA':
            df = df.groupby(["volume", "scene"])
            for group_name, df_group in df:
                # df_group = df_group.reset_index()
                items = df_group.values.tolist()
                self.files += list(combinations(items, 2))
        elif training_mode == "cam" or training_mode == "cam+HA":
            df = df.groupby(["volume", "scene", "cam"])
            for group_name, df_group in df:
                # df_group = df_group.reset_index()
                items = df_group.values.tolist()
                self.files += list(combinations(items, 2))
        elif training_mode == "con" or training_mode == "con+HA":
            df = df.groupby(["volume", "scene", "cam"])
            for group_name, df_group in df:
                # df_group = df_group.reset_index()
                items = df_group.values.tolist()
                items.sort()
                if interval==-1:
                    self.files += list(zip(items, items[1:]))
                    self.files += list(zip(items, items[2:]))
                elif interval==-2:
                    self.files += list(zip(items, items[1:]))
                    self.files += list(zip(items, items[2:]))
                    self.files += list(zip(items, items[3:]))
                else:
                    self.files += list(zip(items, items[interval:]))
        elif training_mode == "HA" or training_mode=='HA_wo_sp':
            self.files = list(zip(df.values.tolist(), df.values.tolist()))
        else:
            raise ValueError('Training mode can only be: HA, scene, cam, con, scene+HA, cam+HA or con+HA')
        if len(self.files) % 8 == 1:
            self.files = self.files[:-1]

    def __len__(self):
        return len(self.files)

    def _read_image(self, volume, scene, cam, frame):
        # h5 = h5py.File(filename, 'r')
        # image = np.array(h5['dataset'][:], dtype='f')
        image = self.data_util_loader.loadBgr(volume, scene, cam, frame)
        image[image>1] = 1
        image = image * 255
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        return image

    def __getitem__(self, idx):

        info_target = self.files[idx][0][:4]
        info_source = self.files[idx][1][:4]
        source_position_map = self.data_util_loader.loadPositionMap(info_source[0], info_source[1], info_source[2], info_source[3])
        target_position_map = self.data_util_loader.loadPositionMap(info_target[0], info_target[1], info_target[2], info_target[3])
        source_reflectance_map = self.data_util_loader.loadReflectance(info_source[0], info_source[1], info_source[2], info_source[3])
        target_reflectance_map = self.data_util_loader.loadReflectance(info_target[0], info_target[1], info_target[2], info_target[3])
        source_R_CW, source_t_CW = self.data_util_loader.loadCamPose(info_source[0], info_source[1], info_source[2], info_source[3])
        target_R_CW, target_t_CW = self.data_util_loader.loadCamPose(info_target[0], info_target[1], info_target[2], info_target[3])
        source_center = torch.tensor([[512],[384]])
        if self.center_crop:
            _, target_center, _ = self.reprojection.warp(source_center,
                                        torch.from_numpy(source_position_map),
                                        torch.from_numpy(target_R_CW), torch.from_numpy(target_t_CW),
                                        mask_fov=True,
                                        mask_occlusion=None,
                                        mask_reflectance=None)
            target_center = target_center.long()
            scenter2tcenter = target_center - source_center
            scenter2tcenter = [scenter2tcenter[0].item(), scenter2tcenter[1].item()]
        else:
            scenter2tcenter = [0, 0]
        
        if self.loading_mode == 'jpg':
            source_name = str(info_source[0]) \
                    + '_' + str(info_source[1]) \
                    + '_' + str(info_source[2]) \
                    + '_' + str(info_source[3])
            target_name = str(info_target[0]) \
                    + '_' + str(info_target[1]) \
                    + '_' + str(info_target[2]) \
                    + '_' + str(info_target[3])
            # print('source: ' + source_name)
            # print('target: ' + target_name)
            image_source = self.data_util_loader.loadRgb(info_source[0], info_source[1], info_source[2], info_source[3])
            image_target = self.data_util_loader.loadRgb(info_target[0], info_target[1], info_target[2], info_target[3])
            image_source = image_source.astype(np.uint8)
            image_target = image_target.astype(np.uint8)
            image_source = Image.fromarray(image_source)
            image_target = Image.fromarray(image_target)
        else: 
            image_source = self._read_image(info_source[0], info_source[1], info_source[2], info_source[3])
            image_target = self._read_image(info_target[0], info_target[1], info_target[2], info_target[3])
        # image_target = crop(image_target, 192, 256, 384, 512)
        if self.center_crop:
            image_target = crop(image_target, 192+scenter2tcenter[1], 256+scenter2tcenter[0], 384, 512)
            image_source = crop(image_source, 192, 256, 384, 512)
        if image_target.mode == 'L':
            image_target_new = Image.new("RGB", image_target.size)
            image_target_new.paste(image_target)
            image_source_new = Image.new("RGB", image_source.size)
            image_source_new.paste(image_target)
            sample = {'image_target': image_target_new, 'image_source': image_source_new, 'idx': idx, 'metainfo': [source_position_map, target_position_map, source_reflectance_map, target_R_CW, target_t_CW], 'source_frame': info_source, 'target_frame': info_target, 'scenter2tcenter': scenter2tcenter, 'metainfo_inv': [target_position_map, source_position_map, target_reflectance_map, source_R_CW, source_t_CW]}
        else:
            sample = {'image_target': image_target, 'image_source': image_source, 'idx': idx, 'metainfo': [source_position_map, target_position_map, source_reflectance_map, target_R_CW, target_t_CW], 'source_frame': info_source, 'target_frame': info_target, 'scenter2tcenter': scenter2tcenter, 'metainfo_inv': [target_position_map, source_position_map, target_reflectance_map, source_R_CW, source_t_CW]}
        # sample = {'image_target': image_target, 'image_source': image_source, 'idx': idx, 'metainfo': [source_position_map, target_position_map, source_reflectance_map, target_R_CW, target_t_CW]}

        if self.data_transform:
            sample = self.data_transform(sample)
        return sample
