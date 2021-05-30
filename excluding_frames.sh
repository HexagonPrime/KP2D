#!/bin/bash -l
#SBATCH --output=test_excl.out
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --constrain='titan_xp|geforce_gtx_titan_x|geforce_rtx_2080_ti|titan_x|titan_xp|'

source /scratch_net/biwidl306/shecai/conda/etc/profile.d/conda.sh
conda activate kp2d
python excluding_frames.py --threshold 8