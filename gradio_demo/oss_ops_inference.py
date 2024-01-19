r""" HyperAverageMetercorrelation Squeeze testing code """

import argparse
import sys
import os
from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image

from segment_anything import SamPredictor, SamAutomaticMaskGenerator

from gradio_demo.Matcher import Matcher
from matcher.common import utils

import random
random.seed(0)


def default_argument_parser():

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Matcher Pytorch Implementation for One-shot Segmentation')

    # Dataset parameters
    parser.add_argument('--datapath', type=str, default='datasets')
    parser.add_argument('--benchmark', type=str, default='coco',
                        choices=['fss', 'coco', 'lvis', 'paco_part', 'pascal_part'])
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=518)
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--log-root', type=str, default='output/coco/fold0')
    parser.add_argument('--visualize', type=int, default=0)

    # DINOv2 and SAM parameters
    parser.add_argument('--dinov2-weights', type=str, default="models/dinov2_vitl14_pretrain.pth")
    parser.add_argument('--sam-weights', type=str, default="models/sam_vit_h_4b8939.pth")
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
    parser.add_argument('--sample-range', type=tuple, default=(4,6), help='sample points number range')
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
    return args

def definite_argument_parser(args, version=1):

    if version==1:

        args.max_sample_iterations = 64
        args.box_nms_thresh = 0.65
        args.sample_range = (1, 6)
        args.topk_scores_threshold = 0.0
        args.use_dense_mask = 1
        args.use_points_or_centers = True
        args.purity_filter = 0.02
        args.iou_filter = 0.85
        args.multimask_output = 1
        args.sel_stability_score_thresh = 0.90
        args.use_score_filter = True
        args.alpha = 1.0
        args.beta = 0.
        args.exp = 0.
        args.num_merging_mask = 9
    elif version == 2:
        args.max_sample_iterations = 30
        args.sample_range = (4, 6)
        args.multimask_output = 0
        args.alpha = 0.8
        args.beta = 0.2
        args.exp = 1.
        args.num_merging_mask = 10
    elif version == 3:
        args.max_sample_iterations = 128
        args.sample_range = (3, 6)
        args.use_box = True
        args.use_points_or_centers = True
        args.coverage_filter = 0.3
        args.alpha = 0.5
        args.beta = 0.5
        args.exp = 0.
        args.num_merging_mask = 5

    return args

def preprocess_data(kwargs, args=None):

    img_size = args.img_size
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor()
    ])

    support_img = Image.fromarray(kwargs.get("support_img"))
    query_img_1 = Image.fromarray(kwargs.get("query_img_1"))
    query_img_2 = Image.fromarray(kwargs.get("query_img_2"))

    support_img_ori_size = (support_img.size[1], support_img.size[0]) # H, W
    query_img_1_ori_size = (query_img_1.size[1], query_img_1.size[0])
    query_img_2_ori_size = (query_img_2.size[1], query_img_2.size[0])


    support_img = transform(support_img)
    query_img_1 = transform(query_img_1)
    query_img_2 = transform(query_img_2)

    support_mask = torch.tensor(kwargs.get("support_mask"))
    support_mask = F.interpolate(support_mask.unsqueeze(0).float(), support_img.size()[-2:],
                               mode='nearest') > 0
    query_imgs = torch.stack([query_img_1, query_img_2], dim=0)

    data = {
        "support_img": support_img[None, ...],
        "support_mask": support_mask,
        "query_imgs": query_imgs,
        "support_img_ori_size": support_img_ori_size,
        "query_imgs_ori_size": (query_img_1_ori_size, query_img_2_ori_size),
    }

    return data

def preprocess_support_mask(data, predictor, version=1):

    if version == 3:
        return data

    sup_mask = data['support_mask'].squeeze()
    H, W = sup_mask.shape[-2:]
    input_points = sup_mask.nonzero().numpy()[:1,::-1]#[:,::-1]
    input_label = np.array([1]*len(input_points))

    support_img_np = data['support_img'].mul(255).byte()
    support_img_np = support_img_np.squeeze().permute(1,2,0).cpu().numpy()

    # forward encoder to obtain image feature
    predictor.reset_image()
    predictor.set_image(support_img_np)

    # mask, _, _ = predictor.predict(
    #     point_coords=input_points,
    #     point_labels=input_label,
    #     multimask_output=False #True
    # )
    mask, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_label,
        multimask_output=True  # True
    )
    predictor.reset_image()

    # show_img_point_box_mask(
    #     support_img_np,
    #     masks=mask,
    #     save_path='test1.png',
    #     mode='mask'
    # )

    # data['support_mask'] = torch.tensor(mask[:1])[None, ...]
    data['support_mask'] = torch.tensor(mask[-1:])[None, ...]

    return data

def main_oss_ops(**kwargs):

    args = default_argument_parser()
    args = definite_argument_parser(args, kwargs.get("version"))

    # Model initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # create sam
    sam = kwargs.get("sam")
    predictor = SamPredictor(sam)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        points_per_batch=64,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        stability_score_offset=1.0,
        sel_stability_score_thresh=args.sel_stability_score_thresh,
        sel_pred_iou_thresh=args.iou_filter,
        box_nms_thresh=args.box_nms_thresh,
        sel_output_layer=args.output_layer,
        output_layer=args.dense_multimask_output,
        dense_pred=args.use_dense_mask,
        multimask_output=args.dense_multimask_output > 0,
        sel_multimask_output=args.multimask_output > 0,
    )

    # create dinov2, large
    dinov2 = kwargs.get("dinov2")

    # create matcher
    score_filter_cfg = {
        "emd": args.emd_filter,
        "purity": args.purity_filter,
        "coverage": args.coverage_filter,
        "score_filter": args.use_score_filter,
        "score": args.deep_score_filter,
        "score_norm": args.deep_score_norm_filter,
        "topk_scores_threshold": args.topk_scores_threshold
    }

    matcher = Matcher(
        encoder=dinov2,
        generator=generator,
        num_centers=args.num_centers,
        use_box=args.use_box,
        use_points_or_centers=args.use_points_or_centers,
        sample_range=args.sample_range,
        max_sample_iterations=args.max_sample_iterations,
        alpha=args.alpha,
        beta=args.beta,
        exp=args.exp,
        score_filter_cfg=score_filter_cfg,
        num_merging_mask=args.num_merging_mask,
        device=args.device
    )

    # process data
    data = preprocess_data(kwargs, args=args)
    data = preprocess_support_mask(data, predictor, version=kwargs.get("version"))

    # inference
    with torch.no_grad():
        utils.fix_randseed(0)
        pred_masks, pred_mask_lists = [], []

        # support mask
        support_img_ori_size = data['support_img_ori_size']
        mask = data['support_mask'].to(predictor.model.device).float()
        mask = F.interpolate(mask, support_img_ori_size, mode="bilinear", align_corners=False) > 0
        mask = mask.squeeze(0).cpu().numpy()
        pred_masks.append(mask)
        pred_mask_lists.append(None)

        for query_img, query_img_ori_size in zip(data['query_imgs'], data['query_imgs_ori_size']):
            data['query_img'], data['query_img_ori_size'] = query_img[None, ...], query_img_ori_size

            support_imgs, support_masks = data["support_img"].to(matcher.device)[None, ...], data["support_mask"].to(matcher.device)  # (1, 1, 3, H, W), (1, 1, H, W)
            query_img, query_img_ori_size = data['query_img'].to(matcher.device), data['query_img_ori_size']  # (1, 3, H, W), img_size

            # 1. Matcher prepare references and target
            matcher.set_reference(support_imgs, support_masks)
            matcher.set_target(query_img, query_img_ori_size)

            # 2. Predict mask of target
            pred_mask, pred_mask_list = matcher.predict()
            matcher.clear()

            pred_masks.append(pred_mask)
            pred_mask_lists.append(pred_mask_list)


    return pred_masks, pred_mask_lists
