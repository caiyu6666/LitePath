#!/bin/bash
#SBATCH -J PathDistill
#SBATCH -o /scratch/timnhaoprj/ycaibt/experiments/PathDistill/slurm_log/job_%j.out
#SBATCH -e /scratch/timnhaoprj/ycaibt/experiments/PathDistill/slurm_log/job_%j.err
#SBATCH --partition=normal
#SBATCH --account=timnhaoprj
#SBATCH -N 1 -n 64 --gres=gpu:8
#SBATCH --time=72:00:00

source /home/ycaibt/.bashrc

cd LitePath/distillation/

/home/ycaibt/anaconda3/envs/torch/bin/python -u extract_features_ddp.py

/home/ycaibt/anaconda3/envs/torch/bin/python merge_features.py
