#!/bin/bash

for fold in 0;
do
  python main_oss.py \
    --benchmark fss \
    --nshot 5 \
    --max_sample_iterations 30 \
    --sample-range "(4,6)" \
    --multimask_output 0 \
    --alpha 0.8 --beta 0.2 --exp 1. \
    --num_merging_mask 5 \
    --fold ${fold} --log-root "output/fss_5shot/fold${fold}"
done
wait
