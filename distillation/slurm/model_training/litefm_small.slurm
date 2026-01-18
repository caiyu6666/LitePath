#!/bin/bash
#SBATCH -J PathDistill
#SBATCH -o /scratch/timnhaoprj/ycaibt/slurm_log/job_%j.out
#SBATCH -e /scratch/timnhaoprj/ycaibt/slurm_log/job_%j.err
#SBATCH --partition=normal
#SBATCH --account=timnhaoprj
#SBATCH -N 1 -n 4 --gres=gpu:4
#SBATCH --time=72:00:00

source /home/ycaibt/.bashrc

cd LitePath/distillation/

exp_name='litefm_small'

log_dir="/scratch/timnhaoprj/ycaibt/experiments/PathDistill/${exp_name}"
mkdir -p $log_dir
log_file="${log_dir}/log.txt"
config_file="configs/${exp_name}.yaml"

/home/ycaibt/anaconda3/envs/torch/bin/python -u train_lite_fast.py --config "$config_file" > "$log_file" 2>&1
