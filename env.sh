#!/bin/bash -l
#SBATCH --output=onegpu.out
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --constrain='titan_xp|geforce_gtx_titan_x|geforce_rtx_2080_ti|tesla_k40c|geforce_gtx_1080_ti'

source /scratch_net/biwidl306/shecai/conda/etc/profile.d/conda.sh
conda activate kp2d
HOROVOD_CUDA_HOME=$CONDA_PREFIX HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod