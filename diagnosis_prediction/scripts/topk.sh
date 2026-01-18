ROOT_WSI="/ssd/Pathology/Patches/Nanfang_Lung_Cohort1/pt_files,/ssd/Pathology/Patches/Nanfang_Lung_Hebeisiyuan/pt_files,/ssd/Pathology/Patches/Nanfang_Lung_Cohort2/pt_files,/ssd/Pathology/Patches/Nanfang_Lung_Cancerous/pt_files,/ssd/Pathology/Patches/Nanfang_Lung_cancerVSbenign/pt_files,/ssd/Pathology/Patches/Nanfang_Lung_Frozen/pt_files,/ssd/Pathology/Patches/Zhefuyi_Lung/pt_files,/ssd/Pathology/Patches/Nanfang_Lung_Prospective/pt_files,/ssd/Pathology/Patches/Nanfang-Lung-Finegrained/pt_files"
studies="Nanfang_primary_metastatic Nanfang-Lung-NSCLC Nanfang_lung_finegrained Nanfang-Lung-Frozen-LymphNodeMetastasis Nanfang_lung_P63"
GPU_LIST="0 1 2 3 4 5 6 7 8"
feature="LiteFM"

seeds=(0 1 2 3 4 5 6 7 8 9)

max_jobs=10  # 最大并行数
job_count=0

for seed in "${seeds[@]}"
do
    for study in $studies
    do
        # 选择最佳GPU
        selected_gpu=-1
        max_free=0
        for gpu_index in $GPU_LIST; do
            free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu_index | awk '{print $1}')
            threshold=5000

            if [ $free_memory -ge $threshold ] && [ $free_memory -gt $max_free ]; then
                selected_gpu=$gpu_index
                max_free=$free_memory
            fi
        done

        if [ $selected_gpu -ne -1 ]; then
            log_dir=logs_topk/${feature}/${study}
            mkdir -p $log_dir

            log_file="${log_dir}/s${seed}.log"

            CUDA_VISIBLE_DEVICES=$selected_gpu python -u main_mil_eval_topk.py --model ABMIL \
                                --study $study \
                                --root ${ROOT_WSI} \
                                --feature $feature \
                                --seed $seed > $log_file 2>&1 &
            sleep 10
            ((job_count++))
            if [[ $job_count -ge $max_jobs ]]; then
                wait -n  # 等待任意一个后台任务完成
                ((job_count--))
            fi
        else
            echo "No available GPU found for study $study with seed $seed"
        fi
    done
done

echo "All jobs have been submitted."
