#!/bin/bash -l
#
#SBATCH -C pronghorn
#SBATCH -A SCSG0002
#SBATCH -J test
#
#SBATCH -n 2
#SBATCH -t 06:00:00
#SBATCH -p dav
#SBATCH -e job_name.err.%J
#SBATCH -o job_name.out.%J

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

source /glade/u/home/joshuadr/tensor_test/bin/activate
module load cuda/9.0

cd /glade/work/joshuadr/SIParCS_ML/algorithm_scripts/

python3 time_accuracy_test.py
