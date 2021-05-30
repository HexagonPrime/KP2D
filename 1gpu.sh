#!/bin/bash -l
#SBATCH --output=%j.out
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --constrain='titan_xp|geforce_gtx_titan_x|geforce_rtx_2080_ti|titan_x|titan_xp|'

source /scratch_net/biwidl306/shecai/conda/etc/profile.d/conda.sh
conda activate kp2d
python train_keypoint_net.py --file kp2d/configs/v4.yaml --training_mode coco --pretrained_model /scratch_net/biwidl306_second/shecai/pretrained_models/v4.ckpt