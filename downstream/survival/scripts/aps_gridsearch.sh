#!/usr/bin/env bash

model="ABMIL"
entry="main_aps_val.py"
tasks=(
    "LUAD_survival"
    "CRC_survival"
    "HNSC_survival"
    "SKCM_survival"
)
feature="LiteFM"
seeds=(1 2 3)
GPU_LIST="0 1 2 3 4 5 6 7"
folds=(0 1 2 3 4)
max_parallel=15

pt_roots=$(cat <<EOF
{
  "TCGA-LUAD": "/home/ycaibt/Pathology/Patches/TCGA-LUAD/pt_files/${feature}/",
  "TCGA-COAD": "/home/ycaibt/Pathology/Patches/TCGA-COAD/pt_files/${feature}/",
  "TCGA-READ": "/home/ycaibt/Pathology/Patches/TCGA-READ/pt_files/${feature}/",
  "TCGA-HNSC": "/home/ycaibt/Pathology/Patches/TCGA-HNSC/pt_files/${feature}/",
  "TCGA-SKCM": "/home/ycaibt/Pathology/Patches/TCGA-SKCM/pt_files/${feature}/"
}
EOF
)

for task in "${tasks[@]}"; do
    log_dir="logs_aps_gridsearch/${feature}/${task}"
    mkdir -p "${log_dir}"
    for seed in "${seeds[@]}"; do
        for fold in "${folds[@]}"; do
            selected_gpu=-1
            max_free=0
            for gpu_index in $GPU_LIST; do
                free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_index" | awk '{print $1}')
                threshold=5000
                if [ "$free_memory" -ge "$threshold" ] && [ "$free_memory" -gt "$max_free" ]; then
                    selected_gpu=$gpu_index
                    max_free=$free_memory
                fi
            done

            if [ "$selected_gpu" -ne -1 ]; then
                log_file="${log_dir}/${seed}_fold${fold}.log"
                echo "$(date): [task: ${task}]-[feature: ${feature}]-[model: ${model}]-[seed: ${seed}]-[fold: ${fold}] -> ${log_file}"
                CUDA_VISIBLE_DEVICES=$selected_gpu python -u "$entry" --model "$model" \
                    --study "$task" \
                    --feature "$feature" \
                    --pt_roots "$pt_roots" \
                    --num_epoch 100 \
                    --temperature 0.7 \
                    --seed "$seed" \
                    --k_start "$fold" \
                    --k_end "$((fold + 1))" > "$log_file" 2>&1 &
                sleep 5
                if (( $(jobs -pr | wc -l) >= max_parallel )); then
                    wait -n
                fi
            else
                echo "No available GPU found for fold $fold with seed $seed"
            fi
        done
    done
done

wait
