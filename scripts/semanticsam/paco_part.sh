#!/bin/bash

python main_oss.py \
  --benchmark paco_part \
  --max_sample_iterations 30 \
  --sample-range "(1,2)" \
  --multimask_output 0 \
  --alpha 0.8 --beta 0.2 --exp 1. \
  --num_merging_mask 10 \
  --num_centers 9 \
  --use_semantic_sam \
  --semantic-sam-weights models/swint_only_sam_many2many.pth \
  --fold 0 --log-root "output/semanticsam/paco/fold0"
