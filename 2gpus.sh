#!/bin/bash -l
#SBATCH --output=twogpus.out
#SBATCH --mem=40G
#SBATCH --gres=gpu:2
#SBATCH --constrain='titan_xp|geforce_gtx_titan_x|geforce_rtx_2080_ti|tesla_k40c|geforce_gtx_1080_ti'
#SBATCH --cpus-per-task=5

source /scratch_net/biwidl306/shecai/conda/etc/profile.d/conda.sh
conda activate kp2d
nvidia-smi -L | wc -l
horovodrun -np 2 -H localhost:1 python train_keypoint_net.py --file kp2d/configs/v4.yaml --mode coco --pretrained_model /scratch_net/biwidl306_second/shecai/pretrained_models/v4.ckpt
# horovodrun -np 2 -H server1:1,server2:1 python train_keypoint_net.py --file kp2d/configs/v4.yaml --mode coco --pretrained_model /scratch_net/biwidl306_second/shecai/pretrained_models/v4.ckpt