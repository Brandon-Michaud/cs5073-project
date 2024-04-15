#!/bin/bash
#
#SBATCH --gres=gpu:1
#SBATCH --partition=disc_dual_a100_students,gpu_a100,gpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=80G
#SBATCH --output=outputs/job_%j_stdout.txt
#SBATCH --error=outputs/job_%j_stderr.txt
#SBATCH --time=00:30:00
#SBATCH --job-name=test
#SBATCH --mail-user=brandondmichaud@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504319/cs5073-project

. /home/fagg/tf_setup.sh
conda activate anne

python base.py -vv @exp.txt @cifar100.txt --cpus_per_task $SLURM_CPUS_PER_TASK