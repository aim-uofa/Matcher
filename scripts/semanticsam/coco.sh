#!/bin/bash


python main_oss.py  \
  --benchmark coco \
  --max_sample_iterations 64 \
  --sample-range "(1,6)" \
  --topk_scores_threshold 0.0 \
  --use_dense_mask 1 \
  --purity_filter 0.02 \
  --iou_filter 0.85 \
  --multimask_output 1 \
  --sel_stability_score_thresh 0.90 \
  --use_score_filter \
  --alpha 1.0 --beta 0. --exp 0. \
  --num_merging_mask 9  \
  --num_centers 7 \
  --use_semantic_sam \
  --semantic-sam-weights models/swint_only_sam_many2many.pth \
  --fold 0 --log-root "output/semanticsam/coco/fold0"
