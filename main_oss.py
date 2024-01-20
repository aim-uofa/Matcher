r""" Matcher testing code for one-shot segmentation """
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('./')

from matcher.common.logger import Logger, AverageMeter
from matcher.common.vis import Visualizer
from matcher.common.evaluation import Evaluator
from matcher.common import utils
from matcher.data.dataset import FSSDataset
from matcher.Matcher import build_matcher_oss

import random
random.seed(0)


def test(matcher, dataloader, args=None):
    r""" Test Matcher """

    # Freeze randomness during testing for reproducibility
    # Follow HSNet
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        batch = utils.to_cuda(batch)
        query_img, query_mask, support_imgs, support_masks = \
            batch['query_img'], batch['query_mask'], \
            batch['support_imgs'], batch['support_masks']

        # 1. Matcher prepare references and target
        matcher.set_reference(support_imgs, support_masks)
        matcher.set_target(query_img)

        # 2. Predict mask of target
        pred_mask = matcher.predict()
        matcher.clear()

        assert pred_mask.size() == batch['query_mask'].size(), \
            'pred {} ori {}'.format(pred_mask.size(), batch['query_mask'].size())

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou, _ = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Matcher Pytorch Implementation for One-shot Segmentation')

    # Dataset parameters
    parser.add_argument('--datapath', type=str, default='datasets')
    parser.add_argument('--benchmark', type=str, default='coco',
                        choices=['fss', 'coco', 'pascal', 'lvis', 'paco_part', 'pascal_part'])
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=518)
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--log-root', type=str, default='output/debug')
    parser.add_argument('--visualize', type=int, default=0)

    # DINOv2 and SAM parameters
    parser.add_argument('--dinov2-size', type=str, default="vit_large")
    parser.add_argument('--sam-size', type=str, default="vit_h")
    parser.add_argument('--dinov2-weights', type=str, default="models/dinov2_vitl14_pretrain.pth")
    parser.add_argument('--sam-weights', type=str, default="models/sam_vit_h_4b8939.pth")
    parser.add_argument('--use_semantic_sam', action='store_true', help='use semantic-sam')
    parser.add_argument('--semantic-sam-weights', type=str, default="models/swint_only_sam_many2many.pth")
    parser.add_argument('--points_per_side', type=int, default=64)
    parser.add_argument('--pred_iou_thresh', type=float, default=0.88)
    parser.add_argument('--sel_stability_score_thresh', type=float, default=0.0)
    parser.add_argument('--stability_score_thresh', type=float, default=0.95)
    parser.add_argument('--iou_filter', type=float, default=0.0)
    parser.add_argument('--box_nms_thresh', type=float, default=1.0)
    parser.add_argument('--output_layer', type=int, default=3)
    parser.add_argument('--dense_multimask_output', type=int, default=0)
    parser.add_argument('--use_dense_mask', type=int, default=0)
    parser.add_argument('--multimask_output', type=int, default=0)

    # Matcher parameters
    parser.add_argument('--num_centers', type=int, default=8, help='K centers for kmeans')
    parser.add_argument('--use_box', action='store_true', help='use box as an extra prompt for sam')
    parser.add_argument('--use_points_or_centers', action='store_true', help='points:T, center: F')
    parser.add_argument('--sample-range', type=str, default="(4,6)", help='sample points number range')
    parser.add_argument('--max_sample_iterations', type=int, default=30)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--beta', type=float, default=0.)
    parser.add_argument('--exp', type=float, default=0.)
    parser.add_argument('--emd_filter', type=float, default=0.0, help='use emd_filter')
    parser.add_argument('--purity_filter', type=float, default=0.0, help='use purity_filter')
    parser.add_argument('--coverage_filter', type=float, default=0.0, help='use coverage_filter')
    parser.add_argument('--use_score_filter', action='store_true')
    parser.add_argument('--deep_score_norm_filter', type=float, default=0.1)
    parser.add_argument('--deep_score_filter', type=float, default=0.33)
    parser.add_argument('--topk_scores_threshold', type=float, default=0.7)
    parser.add_argument('--num_merging_mask', type=int, default=10, help='topk masks for merging')


    args = parser.parse_args()
    args.sample_range = eval(args.sample_range)

    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    Logger.initialize(args, root=args.log_root)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # Model initialization
    if not args.use_semantic_sam:
        matcher = build_matcher_oss(args)
    else:
        from matcher.Matcher_SemanticSAM import build_matcher_oss as build_matcher_semantic_sam_oss
        matcher = build_matcher_semantic_sam_oss(args)

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize)

    # Dataset initialization
    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    # Test Matcher
    with torch.no_grad():
        test_miou, test_fb_iou = test(matcher, dataloader_test, args=args)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')