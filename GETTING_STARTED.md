## Getting Started with Matcher


### Prepare models

Download the model weights of [DINOv2](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth), [SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swint_only_sam_many2many.pth), and organize them as follows.
```
models/
    dinov2_vitl14_pretrain.pth
    sam_vit_h_4b8939.pth
    swint_only_sam_many2many.pth
```


### Test One-shot Semantic Segmentation

You can test one-shot semantic segmentation performance of Matcher on COCO-20<sup>i</sup>, run:

```
python main_oss.py  \
    --benchmark coco \
    --nshot 1 \
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
    --num_merging_mask 9  \
    --fold 0 --log-root "output/coco/fold0"
```

* You can replace `--benchmark coco` with `--benchmark lvis` to test LVIS-92<sup>i</sup>.
* You can replace `--nshot 1` with `--nshot 5` and replace `--num_merging_mask 9` with `--num_merging_mask 5` to test 5-shot performance on COCO-20<sup>i</sup>.
* You can find more commands in `scripts/` for other datasets.

### Gradio Demo

Launch the local demo built with [gradio](https://gradio.app/):
```
python app.py
```

