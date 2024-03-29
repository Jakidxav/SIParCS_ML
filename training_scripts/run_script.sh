#!/bin/bash -l
#
#SBATCH -C caldera
#SBATCH -A SCSG0002
#SBATCH -J rnn30
#
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH -t 06:00:00
#SBATCH -p dav

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

source /glade/u/home/jakidxav/tensor_test/bin/activate
module load cuda/9.0

cd /glade/work/jakidxav/SIParCS_ML/algorithm_scripts/

python3 ML_NN_algorithms.py
