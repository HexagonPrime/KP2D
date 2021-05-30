#!/bin/bash -l
#SBATCH --output=a.out
#SBATCH --mem=40G

source /scratch_net/biwidl306/shecai/conda/etc/profile.d/conda.sh
conda activate pi-gan
python get_hypersim_paths.py