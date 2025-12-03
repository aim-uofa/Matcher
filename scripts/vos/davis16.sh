
CUDA=0
FOLD_NUM=1
FOLD_INDEX=0
SCALE=0.5
MULTIMASK=-1
DECAY_TYPE=linear
DIST_FILTER=1.8
DECAY_RATIO=25
OUTPUT=output/vos/davis16_multi${MULTIMASK}_filter${SCALE}_filter${DIST_FILTER}_decay_${DECAY_TYPE}${DECAY_RATIO}
if [ ! -d ${OUTPUT} ]; then
    mkdir ${OUTPUT}
fi
echo "${OUTPUT} cuda${CUDA} fold${FOLD_INDEX}"
CUDA_VISIBLE_DEVICES=${CUDA} python main_vos.py \
    --dataset D16 \
    --hard_tgt \
    --output ${OUTPUT} \
    --sam-prompt ransac_point_merge \
    --ransac emd_matching_coverage \
    --multimask_output ${MULTIMASK} \
    --use_all_points_in_ransac 0 \
    --input_size "(504,896)" \
    --mask_pool_thres 0.3 \
    --point_keep 0.8 \
    --pos_ratio 5 \
    --emd_filter 0.75 \
    --emd_norm_filter 0.85 \
    --emd_ratio 0.4 \
    --topk 5 \
    --multi_frame 4 \
    --vote_thres 0.5 \
    --fix_last_frame \
    --fold_num ${FOLD_NUM} \
    --fold_index ${FOLD_INDEX} \
    --point_thres_scale ${SCALE} \
    --memory_decay_type ${DECAY_TYPE} \
    --memory_decay_ratio ${DECAY_RATIO} \
    --center_dist_filter ${DIST_FILTER}


python ./matcher_vos/davis2017-evaluation/evaluation_method.py \
  --davis_path /Path/to/DAVIS17/trainval \
  --set val \
  --year 2016 \
  --task semi-supervised \
  --results_path ${OUTPUT}

