#!/bin/bash

EXP_NAME=${1}
METHOD=${2}
EPOCH_NUM=${3}
HERE_checkpoint=${4}
BACKBONE=${5}

echo $*
echo "Ready?! Go!"

if [ $EXP_NAME == "BCSS" ]; then

  for PATCH_SIZE in 512 256; do
    for RATIO in 0.8 0.5; do
      EXP_NAME1=bcss_${PATCH_SIZE}_${RATIO}
      DATA_ROOT=/data/zhongz2/temp_BCSS/bcss_${PATCH_SIZE}_256_${RATIO}_50_False
      SAVE_ROOT=/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801
      python extract_features_patch_retrieval_eval.py \
        --exp_name "${EXP_NAME1}" \
        --method_name "${METHOD}" \
        --patch_label_file ${DATA_ROOT}/patch_label_file.csv \
        --patch_data_path ${DATA_ROOT}/ALL \
        --codebook_semantic ../search/SISH/checkpoints/codebook_semantic.pt \
        --checkpoint ../search/SISH/checkpoints/model_9.pt \
        --save_filename ${SAVE_ROOT}/${EXP_NAME1}_${METHOD}_${EPOCH_NUM}_feats.pkl \
        --HERE_checkpoint ${HERE_checkpoint} \
        --backbone ${BACKBONE}

    done
  done
elif [ $EXP_NAME == "faiss_bins_count_and_size" ]; then

  python extract_features_patch_retrieval_eval.py --action "faiss_bins_count_and_size"

else

  if [ $EXP_NAME == "PanNuke" ]; then
    DATA_ROOT=/data/zhongz2/temp_PanNuke/
    SAVE_ROOT=/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801
  elif [ $EXP_NAME == "NuCLS" ]; then
    DATA_ROOT=/data/zhongz2/temp_NuCLS/
    SAVE_ROOT=/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801
  elif [ $EXP_NAME == "kather100k" ]; then
    DATA_ROOT=/data/zhongz2/temp_kather100k
    SAVE_ROOT=/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801
  fi
  echo ${EXP_NAME} ${DATA_ROOT} ${SAVE_ROOT}

  python extract_features_patch_retrieval_eval.py \
    --exp_name "${EXP_NAME}" \
    --method_name "${METHOD}" \
    --patch_label_file ${DATA_ROOT}/patch_label_file.csv \
    --patch_data_path ${DATA_ROOT}/ALL \
    --codebook_semantic ../search/SISH/checkpoints/codebook_semantic.pt \
    --checkpoint ../search/SISH/checkpoints/model_9.pt \
    --save_filename ${SAVE_ROOT}/${EXP_NAME}_${METHOD}_${EPOCH_NUM}_feats.pkl \
    --HERE_checkpoint ${HERE_checkpoint} \
    --backbone ${BACKBONE}
fi

exit
