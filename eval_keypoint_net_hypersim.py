# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Example usage: python scripts/eval_keypoint_net.sh --pretrained_model /data/models/kp2d/v4.pth --input_dir /data/datasets/kp2d/HPatches/

import argparse
import os
import pickle
import random
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# from kp2d.datasets.patches_dataset import PatchesDataset
from kp2d.datasets.hypersim import HypersimLoader
from kp2d.evaluation_hypersim.evaluate import evaluate_keypoint_net_hypersim
from kp2d.networks.keypoint_net import KeypointNet
from kp2d.networks.keypoint_resnet import KeypointResnet

from kp2d.datasets.augmentations import to_tensor_sample


def main():
    parser = argparse.ArgumentParser(
        description='Script for KeyPointNet testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pretrained_model", type=str, help="pretrained model path")
    parser.add_argument("--input_dir", required=True, type=str, help="Folder containing input images")
    parser.add_argument("--testing_mode", type=str)

    args = parser.parse_args()
    checkpoint = torch.load(args.pretrained_model)
    model_args = checkpoint['config']['model']['params']

    # # Check model type
    # if 'keypoint_net_type' in checkpoint['config']['model']['params']:
    #     net_type = checkpoint['config']['model']['params']
    # else:
    #     net_type = KeypointNet # default when no type is specified
    net_type = KeypointNet # default when no type is specified

    # Create and load keypoint net
    if net_type is KeypointNet:
        keypoint_net = KeypointNet(use_color=model_args['use_color'],
                                do_upsample=model_args['do_upsample'],
                                do_cross=model_args['do_cross'])
    else:
        keypoint_net = KeypointResnet()

    keypoint_net.load_state_dict(checkpoint['state_dict'])
    keypoint_net = keypoint_net.cuda()
    keypoint_net.eval()
    print('Loaded KeypointNet from {}'.format(args.pretrained_model))
    print('KeypointNet params {}'.format(model_args))

    eval_params = [{'res': (1024, 768), 'top_k': 1000, }]

    for params in eval_params:
        center_crop = False
        if params['res']==(512, 384):
            center_crop=True
            hp_dataset = HypersimLoader(args.input_dir, training_mode='con', data_transform=to_tensor_sample, partition='val+test', interval=1)
        else:
            hp_dataset = HypersimLoader(args.input_dir, training_mode='con', center_crop=False, data_transform=to_tensor_sample, interval=1, partition='val+test')
        data_loader = DataLoader(hp_dataset,
                                 batch_size=1,
                                 pin_memory=False,
                                 shuffle=False,
                                 num_workers=8,
                                 worker_init_fn=None,
                                 sampler=None)

        print(colored('Evaluating for {} -- top_k {}'.format(params['res'], params['top_k']),'green'))
        rep, loc, mscore = evaluate_keypoint_net_hypersim(
            data_loader,
            keypoint_net,
            output_shape=params['res'],
            top_k=params['top_k'],
            use_color=True,
            center_crop=center_crop)

        print('Repeatability {0:.3f}'.format(rep))
        print('Localization Error {0:.3f}'.format(loc))
        print('MScore {:.3f}'.format(mscore))


if __name__ == '__main__':
    main()
