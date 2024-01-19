#!/bin/bash

for fold in 0 1 2 3 4 5 6 7 8 9;
do
  python main_oss.py  \
    --benchmark lvis \
    --nshot 5 \
    --max_sample_iterations 64 \
    --box_nms_thresh 0.65 \
    --sample-range "(1,6)" \
    --topk_scores_threshold 0.0 \
    --use_dense_mask 1 \
    --use_points_or_centers \
    --purity_filter 0.02 \
    --iou_filter 0.85 \
    --multimask_output 1 \
    --sel_stability_score_thresh 0.90 \
    --use_score_filter \
    --alpha 1.0 --beta 0. --exp 0. \
    --num_merging_mask 5 \
     --fold ${fold} --log-root "output/lvis_5shot/fold${fold}"
done
wait