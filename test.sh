#!/bin/bash
#
#SBATCH --partition=disc_dual_a100_students,debug,debug_5min
#SBATCH --cpus-per-task=16
#SBATCH --mem=1G
#SBATCH --output=outputs/job_%j_stdout.txt
#SBATCH --error=outputs/job_%j_stderr.txt
#SBATCH --time=00:05:00
#SBATCH --job-name=test
#SBATCH --mail-user=brandondmichaud@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504319/cs5073-project

. /home/fagg/tf_setup.sh
conda activate anne

python base.py -vv @cifar100_resnet50.txt @oscer.txt --cpus_per_task $SLURM_CPUS_PER_TASK