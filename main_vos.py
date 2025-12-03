from os import path
import os
import argparse
import shutil
import queue
import math
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

import sys
sys.path.append('./')

from matcher_vos.inference.data.test_datasets_nonorm import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from matcher_vos.inference.data.mask_mapper import MaskMapper

import matcher_vos.ddp_utils as ddp_utils

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger

matplotlib_axes_logger.setLevel('ERROR')
import torchshow as ts
import cv2
import ot
import heapq

from matcher_vos.common.logger import Logger, AverageMeter
from matcher_vos.common.vis import Visualizer
from matcher_vos.common import utils
from segment_anything import sam_model_registry, SamPredictor
from dinov2.models import vision_transformer as vits
import dinov2.utils.utils as dinov2_utils
import random
random.seed(0)

from matcher_vos.matcher_new import Matcher


def db_eval_iou(annotation, segmentation):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
    Return:
        jaccard (float): region similarity
 """

    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if annotation.shape[-1] != segmentation.shape[-1] or annotation.shape[0] != segmentation.shape[0]:
        segmentation = Image.fromarray(segmentation.astype(np.uint8))
        h, w = annotation.shape
        segmentation = segmentation.resize((w, h), Image.NEAREST)
        segmentation = np.array(segmentation)
    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
               np.sum((annotation | segmentation), dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default='./models/incontext_v13_99ep.pth')
    parser.add_argument('--model', default='uip_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--update_prompt', action='store_true')
    # parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--input_size', type=str, default="(518, 518)")

    # Data options
    parser.add_argument('--dataset', help='D16/D17/Y18/Y19/LV1/LV3/G', default='D17')
    parser.add_argument('--d16_path', default='/home/yangliu/workspace/pycode/project/SAM/dinov2/data/DAVIS16')
    parser.add_argument('--d17_path', default='/share/project/zxs/datasets/DAVIS17')
    parser.add_argument('--y18_path', default='/share/project/zxs/datasets/YouTubeVOS18')
    parser.add_argument('--y19_path', default='/share/project/zxs/datasets/YouTubeVOS')
    parser.add_argument('--lv_path', default='../long_video_set')
    parser.add_argument('--mose_path', default='/share/project/zxs/datasets/MOSE')

    parser.add_argument('--split', help='val/test', default='val')
    parser.add_argument('--output', default=None)
    parser.add_argument('--save_all', action='store_true',
                        help='Save all frames. Useful only in YouTubeVOS/long-time video', )

    parser.add_argument('--fast_eval', action='store_true')
    parser.add_argument('--all_frame', action='store_true')
    parser.add_argument('--num_frame', default=8, type=int)
    parser.add_argument('--hard_tgt', action='store_true')
    parser.add_argument('--reweight', action='store_true')
    parser.add_argument('--save_scores', action='store_true')
    parser.add_argument('--score_th', default=0.5, type=float)
    parser.add_argument('--resume', default=False, action='store_true')

    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--size', default=518, type=int,
                        help='Resize the shorter side to this size. -1 to use original resolution. ')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--pos_ratio', default=0.0, type=float)
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--box_extend', default=1.0, type=float)
    parser.add_argument('--point_keep', default=0.5, type=float)
    parser.add_argument('--neg_label', default=None, type=str)
    parser.add_argument('--multimask_output', type=int, default=0, help='use multimask_output')
    parser.add_argument('--nan_process_type', type=int, default=0)
    parser.add_argument('--ransac_thres', type=float, default=0.9)
    parser.add_argument('--mask_pool_thres', type=float, default=0.0)
    parser.add_argument('--emd_ratio', type=float, default=1.0)
    parser.add_argument('--multi_frame', type=int, default=1)
    parser.add_argument('--vote_thres', type=float, default=0.5)
    parser.add_argument('--ransac_sample_num', type=int, default=300)
    parser.add_argument('--memory_decay_ratio', type=float, default=20)
    parser.add_argument('--memory_decay_type', default='cos', type=str)
    parser.add_argument('--fix_last_frame', default=False, action='store_true')
    parser.add_argument('--fold_num', type=int, default=1)
    parser.add_argument('--fold_index', type=int, default=0)
    parser.add_argument('--fold_range', type=str, default=None)

    parser.add_argument('--sam-prompt', type=str, default='ransac_point_merge',
                        choices=['point', 'box', 'ransac_point_merge', 'point_box_together', 'coarse mask'])
    parser.add_argument('--ransac', type=str, default='emd_matching',
                        choices=['none', 'global_matching', 'local_matching', 'combined_matching',
                                 'local_matching_coverage', 'combined_matching_coverage', 'local_matching_emd',
                                 'emd_matching', 'emd_matching_coverage', 'coverage_matching'])
    parser.add_argument('--ransac-range', type=str, default="(1,6)", help='ransac points number range')
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--topk', type=int, default=10, help='topk_to_merge_after_ransac')
    parser.add_argument('--K', type=int, default=8, help='K centers for kmeans')
    parser.add_argument('--use_center', type=int, default=1, help='use center instead of point')
    parser.add_argument('--use_negative_label', type=int, default=0, help='use negative label')
    parser.add_argument('--filter', type=int, default=0, help='do_filter')
    parser.add_argument('--topk_scores_threshold', type=int, default=1, help='do_filter_after_ransac')
    parser.add_argument('--use_box_in_ransac', type=int, default=0, help='use_box_in_ransac')
    parser.add_argument('--use_coarse_mask_in_ransac', type=int, default=0, help='use_coarse_mask_in_ransac')
    parser.add_argument('--use_all_points_in_ransac', type=int, default=1, help='use_all_points_in_ransac')
    parser.add_argument('--expand_box', type=int, default=0, help='expand box due to low resolution')
    parser.add_argument('--point_thres_scale', type=float, default=12.0,
                        help='point_thres_scale,less means filter more')
    parser.add_argument('--center_dist_filter', type=float, default=2.0,
                        help='point_thres_scale,less means filter more')
    parser.add_argument('--filtered_before_ransac', type=int, default=0, help='filtered_before_ransac')
    parser.add_argument('--use_negative_point_in_ransac', type=int, default=0, help='num of negative point in ransac')
    parser.add_argument('--coverage_filter', type=float, default=0.0, help='use coverage_filter')
    parser.add_argument('--iou_filter', type=float, default=0.0, help='use iou_filter')
    parser.add_argument('--emd_filter', type=float, default=0.0, help='use emd_filter')
    parser.add_argument('--emd_norm_filter', type=float, default=0.0, help='use emd_norm_filter')

    args = parser.parse_args()
    args.ransac_range = eval(args.ransac_range)
    args.input_size = eval(args.input_size)
    if args.fold_range:
        args.fold_range = eval(args.fold_range)
    args = ddp_utils.init_distributed_mode(args)
    assert not (args.fast_eval and args.distributed)
    if args.multi_frame:
        args.multi_frame -= 1
        if args.fix_last_frame:
            args.multi_frame -= 1

    max_vid = 1000000000000000000000000
    if args.fast_eval:
        max_vid = 20

    if args.output is None:
        args.output = f'../output/painter_vos_{args.dataset}_{args.split}'
        print(f'Output path not provided. Defaulting to {args.output}')

    vis_path = path.join(args.output, "vis")
    visualize = args.visualize

    """
    Data preparation
    """
    is_youtube = args.dataset.startswith('Y')
    is_davis = args.dataset.startswith('D')
    is_lv = args.dataset.startswith('LV')
    is_mose = args.dataset.startswith('MOSE')

    if is_youtube:
        out_path = path.join(args.output, 'Annotations')
    else:
        out_path = args.output

    if is_youtube:
        if args.dataset == 'Y18':
            yv_path = args.y18_path
        elif args.dataset == 'Y19':
            yv_path = args.y19_path

        if args.split == 'val':
            args.split = 'valid'
            meta_dataset = YouTubeVOSTestDataset(data_root=yv_path, split='valid', size=args.size)
        elif args.split == 'test':
            meta_dataset = YouTubeVOSTestDataset(data_root=yv_path, split='test', size=args.size)
        elif args.split == 'train':
            meta_dataset = YouTubeVOSTestDataset(data_root=yv_path, split='train', size=args.size)
        else:
            raise NotImplementedError

    elif is_davis:
        if args.dataset == 'D16':
            if args.split == 'val':
                # Set up Dataset, a small hack to use the image set in the 2017 folder because the 2016 one is of a different format
                meta_dataset = DAVISTestDataset(args.d16_path,
                                                imset='/home/yangliu/workspace/pycode/project/SAM/dinov2/data/DAVIS17/trainval/ImageSets/2016/val.txt',
                                                size=args.size)
            else:
                raise NotImplementedError
            palette = None
        elif args.dataset == 'D17':
            if args.split == 'val':
                if args.fold_range:
                    video_indices = args.fold_range
                else:
                    video_num = 30
                    fold_split = [math.ceil(30 / args.fold_num) * i for i in range(args.fold_num)] + [video_num]
                    video_indices = (fold_split[args.fold_index], fold_split[args.fold_index + 1])
                meta_dataset = DAVISTestDataset(path.join(args.d17_path, 'trainval'), imset='2017/val.txt',
                                                size=args.size, indices=video_indices)
            elif args.split == 'test':
                meta_dataset = DAVISTestDataset(path.join(args.d17_path, 'test-dev'), imset='2017/test-dev.txt',
                                                size=args.size)
            else:
                raise NotImplementedError

    elif is_mose:
        if args.dataset == 'MOSE':
            if args.split == 'val':
                meta_dataset = LongTestDataset(path.join(args.mose_path, 'val'))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    elif is_lv:
        if args.dataset == 'LV1':
            meta_dataset = LongTestDataset(path.join(args.lv_path, 'long_video'))
        elif args.dataset == 'LV3':
            meta_dataset = LongTestDataset(path.join(args.lv_path, 'long_video_x3'))
        else:
            raise NotImplementedError
    elif args.dataset == 'G':
        meta_dataset = LongTestDataset(path.join(args.generic_path), size=args.size)
        if not args.save_all:
            args.save_all = True
            print('save_all is forced to be true in generic evaluation mode.')
    else:
        raise NotImplementedError

    meta_loader = meta_dataset.get_datasets()
    sampler_val = DistributedSampler(meta_dataset, shuffle=False) \
        if args.distributed else SequentialSampler(meta_dataset)
    data_loader_val = DataLoader(
        meta_dataset, batch_size=1, sampler=sampler_val, drop_last=False, num_workers=0,
        collate_fn=ddp_utils.dummy_collate_fn,
    )

    torch.autograd.set_grad_enabled(False)
    # model, model_without_ddp = prepare_model(args.ckpt_path, args.model)
    # model.eval()
    print('Model loaded.')
    device = torch.device("cuda")

    total_process_time = 0
    total_frames = 0

    J_score = []
    data_root = '/share/project/zxs/datasets/YouTubeVOS/train/Annotations'

    # Model initialization
    #### create sam
    sam_checkpoint = "/home/yangliu/workspace/pycode/project/SAM/dinov2/Models/SAM/sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    #### create dinov2, large
    dinov2_weights = "/home/yangliu/workspace/pycode/project/SAM/dinov2/Models/dinov2/dinov2_vitl14_pretrain.pth"
    dinov2_kwargs = dict(
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    dino = vits.__dict__["vit_large"](**dinov2_kwargs)
    dinov2_utils.load_pretrained_weights(dino, dinov2_weights, "teacher")
    dino.eval()
    dino.cuda()
    dino.to(device=device)

    # # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    Visualizer.initialize(True)

    matcher = Matcher(
        encoder=dino,
        predictor=predictor,
        input_size=args.input_size,
        memory_len=4,
        multimask_output=args.multimask_output,
        fix_last_frame=args.fix_last_frame,
        pos_embed_ratio=5,
        visualize=args.visualize,
        vis_path=args.output,
        point_num_thres_scale=args.point_thres_scale,
        center_dist_filter=args.center_dist_filter,
        memory_decay_ratio=args.memory_decay_ratio,
        memory_decay_type=args.memory_decay_type
    )

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)

    eval_video_names = None
    # eval_video_names = ['dogs-jump']

    input_size_ori = args.input_size

    # Start eval
    for i, vid_reader in tqdm(zip(range(max_vid), data_loader_val), total=min(max_vid, len(data_loader_val)),
                              disable=not ddp_utils.is_main_process()):
        vid_reader = vid_reader[0]
        loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)
        vid_name = vid_reader.vid_name
        vid_length = len(loader)

        # if i < 18:
        #     continue

        if eval_video_names and vid_name not in eval_video_names:
            continue

        if args.resume:
            all_preds_exist = True
            for ti, data in enumerate(loader):
                info = data['info']
                frame = info['frame'][0]
                if (args.all_frame or info['save'][0]):
                    pred_path = os.path.join(out_path, vid_name, frame[:-4] + ".png")
                    if not os.path.exists(pred_path):
                        all_preds_exist = False
                        break
            if all_preds_exist:
                os.system(f'echo "video {vid_name} results already exist, skip this video."')
                continue

        mapper = MaskMapper()
        prompt, prompt_target = None, None
        first_mask_loaded = False
        memo, memo_target = None, None
        offset = 0
        img_queue, feat_queue, mask_queue, mask_score_queue = None, None, None, None

        for ti, data in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=True):
                rgb = data['rgb'].cuda()[0]
                msk = data.get('mask')
                info = data['info']
                frame = info['frame'][0]
                shape = info['shape']
                need_resize = info['need_resize'][0]

                frame_id = int(frame.strip(".jpg"))
                # if int(frame.strip(".jpg")) <  45:
                #     continue

                # if not (args.all_frame or info['save'][0] or ti == len(loader) - 1):
                if not (args.all_frame or info['save'][0]):
                    continue

                """
                For timing see https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
                Seems to be very similar in testing as my previous timing method 
                with two cuda sync + time.time() in STCN though 
                """
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                if not first_mask_loaded:
                    if msk is not None:
                        first_mask_loaded = True
                        if msk.shape[1] < msk.shape[2]:
                            args.input_size = input_size_ori
                        else:
                            args.input_size = (input_size_ori[1], input_size_ori[0])
                    else:
                        # no point to do anything without a mask
                        continue

                if args.flip:
                    rgb = torch.flip(rgb, dims=[-1])
                    msk = torch.flip(msk, dims=[-1]) if msk is not None else None

                # Map possibly non-continuous labels to continuous ones
                if msk is not None:
                    msk, labels = mapper.convert_mask(msk[0].numpy(), exhaustive=args.split == 'train')
                    msk = torch.Tensor(msk).cuda()
                    if need_resize:
                        msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                else:
                    labels = None

                # print(i, rgb.shape, msk.shape if msk is not None else None)
                # rgb = F.interpolate(rgb.unsqueeze(0), (args.input_size, args.input_size), mode='bicubic', align_corners=False)[0]
                rgb = F.interpolate(rgb.unsqueeze(0), args.input_size, mode='bicubic', align_corners=False)[0]
                # Run the model on this frame
                if msk is not None:
                    # msk = F.interpolate(msk.unsqueeze(0), (args.input_size, args.input_size), mode='nearest')[0]
                    msk = F.interpolate(msk.unsqueeze(0), args.input_size, mode='nearest')[0]
                    # if prompt is not None:
                    #     memo.insert(0, prompt)
                    #     for cls in range(len(memo_target)):
                    #         memo_target[cls].insert(0, prompt_target[cls])
                    #     offset += 1
                    prompt, prompt_target = rgb, msk
                    matcher.set_reference(rgb[None, ...], msk[None, ...])
                    if memo is None:
                        memo, memo_target = [], [[] for cls_id in range(len(prompt_target))]

                        frame_heapq = []
                        multi_frame_mask_score = []
                        multi_frame_img_np = []
                        multi_frame_feat = []
                        multi_frame_mask = []
                        multi_frame_id = []
                    else:
                        for _ in range(len(memo_target), len(prompt_target)):
                            memo_target.append([])

                support_imgs = prompt[None, None, ...]
                support_masks = prompt_target[None, ...]
                query_img = rgb[None, ...]
                query_mask = None

                tgt = {
                    'tgt_img': query_img,
                    'tgt_frame_id': ti,
                    'tgt_name': vid_name
                }

                pred_masks, pred_masks_sem, mask_scores = matcher.predict(tgt)

                # Upsample to original size if needed
                if need_resize:
                    # prob = F.interpolate(prob.unsqueeze(0), shape, mode='nearest')[0]
                    pred_masks = F.interpolate(pred_masks[None, ...], shape, mode='nearest')[0]
                    pred_masks_sem = F.interpolate(pred_masks_sem[None, None, ...], shape, mode='nearest')[0, 0]
                if args.flip:
                    # prob = torch.flip(prob, dims=[-1])
                    pred_masks = torch.flip(pred_masks, dims=[-1])
                    pred_masks_sem = torch.flip(pred_masks_sem, dims=[-1])

                end.record()
                torch.cuda.synchronize()
                total_process_time += (start.elapsed_time(end) / 1000)
                total_frames += 1

                # Probability mask -> index mask
                # out_mask = torch.argmax(prob, dim=0)
                # out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
                out_mask = pred_masks_sem.cpu().numpy().astype(np.uint8)

                # Save the mask
                if args.save_all or info['save'][0]:
                    this_out_path = path.join(out_path, vid_name)
                    os.makedirs(this_out_path, exist_ok=True)
                    out_mask = mapper.remap_index_mask(out_mask)
                    out_img = Image.fromarray(out_mask)
                    if vid_reader.get_palette() is not None:
                        out_img.putpalette(vid_reader.get_palette())
                    out_img.save(os.path.join(this_out_path, frame[:-4] + '.png'))

                if args.save_scores:
                    prob = prob / (prob.max() + 0.01)
                    prob = (prob.detach().cpu().numpy() * 255).astype(np.uint8)
                    np_path = path.join(args.output, 'Scores', vid_name)
                    os.makedirs(np_path, exist_ok=True)
                    if ti == len(loader) - 1:
                        hkl.dump(mapper.remappings, path.join(np_path, f'backward.hkl'), mode='w')
                    if args.save_all or info['save'][0]:
                        hkl.dump(prob, path.join(np_path, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')

                if args.fast_eval and info['save'][0]:
                    anno = np.array(Image.open(os.path.join(data_root, vid_name, frame[:-4] + '.png')))
                    for i in range(1, anno.max() + 1):
                        J_score.append(db_eval_iou(np.equal(anno, i), np.equal(out_img, i)))

    if args.fast_eval:
        J_score = sum(J_score) / len(J_score)
        print(f'Fast evaluation J score: {J_score}')

    if ddp_utils.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    if ddp_utils.is_main_process():
        print(f'Total processing time: {total_process_time}')
        print(f'Total processed frames: {total_frames}')
        print(f'FPS: {total_frames / total_process_time}')
        print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2 ** 20)}')

        if not (args.save_scores or args.fast_eval):
            if is_youtube:
                print('Making zip for YouTubeVOS...')
                shutil.make_archive(path.join(args.output, path.basename(args.output)), 'zip', args.output,
                                    'Annotations')
            elif is_davis and args.split == 'test' or is_mose:
                print('Making zip for DAVIS test-dev...')
                shutil.make_archive(args.output, 'zip', args.output)
            print('Finished')
