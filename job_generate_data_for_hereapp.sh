#!/bin/bash


ONLY_STEP1=${1}
GROUP_NAME=${2}

SAVE_ROOT="/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backboneCONCH_dropout0.25/analysis/ST_20240903"
python test_deployment_shared_attention_two_images_comparison_v42.py \
    --save_root ${SAVE_ROOT} \
    --svs_dir "/data/zhongz2/ST_20240903/svs" \
    --patches_dir "/data/zhongz2/ST_20240903/patches" \
    --image_ext ".svs" \
    --backbone "CONCH" \
    --ckpt_path "/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backboneCONCH_dropout0.25/split_3/snapshot_53.pt" \
    --cluster_feat_name "feat_before_attention_feat" \
    --csv_filename "/data/zhongz2/ST_20240903/all_20240907.xlsx" \
    --cluster_task_name ${GROUP_NAME} \
    --cluster_task_index 0 \
    --num_patches 128 \
    --only_step1 ${ONLY_STEP1} \
    --pre_computed_vst True

exit;







