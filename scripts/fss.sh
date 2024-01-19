#!/bin/bash

for fold in 0;
do
  python main_oss.py \
    --benchmark fss \
    --max_sample_iterations 30 \
    --sample-range "(4,6)" \
    --multimask_output 0 \
    --alpha 0.8 --beta 0.2 --exp 1. \
    --num_merging_mask 10 \
     --fold ${fold} --log-root "output/fss/fold${fold}"
done
wait