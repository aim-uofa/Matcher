#!/bin/bash

for fold in 0 1 2 3;
do
  python  main_oss.py \
    --benchmark paco_part \
    --max_sample_iterations 128 \
    --sample-range "(3,6)" \
    --use_box \
    --use_points_or_centers \
    --coverage_filter 0.3 \
    --alpha 0.5 --beta 0.5 --exp 0. \
    --num_merging_mask 5 \
    --fold ${fold}  --log-root "output/paco/fold${fold}"
done
wait
