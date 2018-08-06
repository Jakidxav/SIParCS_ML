#!/bin/bash -l
#
#BATCH -C calderaTCH -C caldera
#SBATCH -A SCSG0002
#SBATCH -J rnn20
#
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH -t 06:00:00
#SBATCH -p dav

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

source /glade/u/home/joshuadr/tensor_test/bin/activate
module load cuda/9.0

cd /glade/work/joshuadr/SIParCS_ML/keras/algorithm_scripts/

python3 rnn_20.py
