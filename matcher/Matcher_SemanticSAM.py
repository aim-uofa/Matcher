import os
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import cv2
import ot
import math
from scipy.optimize import linear_sum_assignment

from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from dinov2.models import vision_transformer as vits
import dinov2.utils.utils as dinov2_utils
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform

from matcher.k_means import kmeans_pp

from semantic_sam import prepare_image, plot_multi_results, build_semantic_sam, SemanticSAMPredictor

import random

class Matcher:
    def __init__(
            self,
            encoder,
            generator=None,
            input_size=518,
            num_centers=8,
            use_box=False,
            use_points_or_centers=True,
            sample_range=(4, 6),
            max_sample_iterations=30,
            alpha=1.,
            beta=0.,
            exp=0.,
            score_filter_cfg=None,
            num_merging_mask=10,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        # models
        self.encoder = encoder
        self.generator = generator
        self.rps = None

        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.input_size = input_size

        # transforms for image encoder
        self.encoder_transform = transforms.Compose([
            MaybeToTensor(),
            make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.tar_img = None
        self.tar_images = None
        self.tar_img_np = None

        self.ref_imgs = None
        self.ref_masks_pool = None
        self.nshot = None

        self.encoder_img_size = None
        self.encoder_feat_size = None

        self.num_centers = num_centers
        self.use_box = use_box
        self.use_points_or_centers = use_points_or_centers
        self.sample_range = sample_range
        self.max_sample_iterations =max_sample_iterations

        self.alpha, self.beta, self.exp = alpha, beta, exp
        assert score_filter_cfg is not None
        self.score_filter_cfg = score_filter_cfg
        self.num_merging_mask = num_merging_mask

        self.device = device

    def set_reference(self, imgs, masks):

        def reference_masks_verification(masks):
            if masks.sum() == 0:
                _, _, sh, sw = masks.shape
                masks[..., (sh // 2 - 7):(sh // 2 + 7), (sw // 2 - 7):(sw // 2 + 7)] = 1
            return masks

        imgs = imgs.flatten(0, 1)  # bs, 3, h, w
        img_size = imgs.shape[-1]
        assert img_size == self.input_size[-1]
        feat_size = img_size // self.encoder.patch_size

        self.encoder_img_size = img_size
        self.encoder_feat_size = feat_size

        # process reference masks
        masks = reference_masks_verification(masks)
        masks = masks.permute(1, 0, 2, 3)  # ns, 1, h, w
        ref_masks_pool = F.avg_pool2d(masks, (self.encoder.patch_size, self.encoder.patch_size))
        nshot = ref_masks_pool.shape[0]
        ref_masks_pool = (ref_masks_pool > 0.).float()
        ref_masks_pool = ref_masks_pool.reshape(-1)  # nshot, N

        self.ref_imgs = imgs
        self.ref_masks_pool = ref_masks_pool
        self.nshot = nshot

    def set_target(self, img):

        img_h, img_w = img.shape[-2:]
        assert img_h == self.input_size[0] and img_w == self.input_size[1]

        # transform query to numpy as input of sam
        img_np = img.mul(255).byte()
        img_np = img_np.squeeze(0).permute(1, 2, 0).cpu().numpy() # H, W, 3

        from PIL import Image

        img_pil = Image.fromarray(img_np.astype('uint8')).convert('RGB')
        t = []
        t.append(transforms.Resize(640, interpolation=Image.BICUBIC))
        transform1 = transforms.Compose(t)
        image_ori = transform1(img_pil)

        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()

        self.tar_img = img
        self.tar_images = images
        self.tar_img_np = image_ori

    def set_rps(self):
        if self.rps is None:
            assert self.encoder_feat_size is not None
            self.rps = RobustPromptSampler(
                encoder_feat_size=self.encoder_feat_size,
                sample_range=self.sample_range,
                max_iterations=self.max_sample_iterations
            )


    def predict(self):

        ref_feats, tar_feat = self.extract_img_feats()
        all_points, box, S, C, reduced_points_num = self.patch_level_matching(ref_feats=ref_feats, tar_feat=tar_feat)
        points = self.clustering(all_points) if not self.use_points_or_centers else all_points
        self.set_rps()
        pred_masks = self.mask_generation(self.tar_img_np, points, box, all_points, self.ref_masks_pool, C)
        return pred_masks


    def extract_img_feats(self):

        ref_imgs = torch.cat([self.encoder_transform(rimg)[None, ...] for rimg in self.ref_imgs], dim=0)
        tar_img = torch.cat([self.encoder_transform(timg)[None, ...] for timg in self.tar_img], dim=0)

        ref_feats = self.encoder.forward_features(ref_imgs.to(self.device))["x_prenorm"][:, 1:]
        tar_feat = self.encoder.forward_features(tar_img.to(self.device))["x_prenorm"][:, 1:]
        # ns, N, c = ref_feats.shape
        ref_feats = ref_feats.reshape(-1, self.encoder.embed_dim)  # ns*N, c
        tar_feat = tar_feat.reshape(-1, self.encoder.embed_dim)  # N, c

        ref_feats = F.normalize(ref_feats, dim=1, p=2) # normalize for cosine similarity
        tar_feat = F.normalize(tar_feat, dim=1, p=2)

        return ref_feats, tar_feat

    def patch_level_matching(self, ref_feats, tar_feat):

        # forward matching
        S = ref_feats @ tar_feat.t()  # ns*N, N
        C = (1 - S) / 2  # distance

        S_forward = S[self.ref_masks_pool.flatten().bool()]

        indices_forward = linear_sum_assignment(S_forward.cpu(), maximize=True)
        indices_forward = [torch.as_tensor(index, dtype=torch.int64, device=self.device) for index in indices_forward]
        sim_scores_f = S_forward[indices_forward[0], indices_forward[1]]
        indices_mask = self.ref_masks_pool.flatten().nonzero()[:, 0]

        # reverse matching
        S_reverse = S.t()[indices_forward[1]]
        indices_reverse = linear_sum_assignment(S_reverse.cpu(), maximize=True)
        indices_reverse = [torch.as_tensor(index, dtype=torch.int64, device=self.device) for index in indices_reverse]
        retain_ind = torch.isin(indices_reverse[1], indices_mask)
        if not (retain_ind == False).all().item():
            indices_forward = [indices_forward[0][retain_ind], indices_forward[1][retain_ind]]
            sim_scores_f = sim_scores_f[retain_ind]
        inds_matched, sim_matched = indices_forward, sim_scores_f

        reduced_points_num = len(sim_matched) // 2 if len(sim_matched) > 40 else len(sim_matched)
        sim_sorted, sim_idx_sorted = torch.sort(sim_matched, descending=True)
        sim_filter = sim_idx_sorted[:reduced_points_num]
        points_matched_inds = indices_forward[1][sim_filter]

        points_matched_inds_set = torch.tensor(list(set(points_matched_inds.cpu().tolist())))
        points_matched_inds_set_w = points_matched_inds_set % (self.encoder_feat_size)
        points_matched_inds_set_h = points_matched_inds_set // (self.encoder_feat_size)
        idxs_mask_set_x = (points_matched_inds_set_w * self.encoder.patch_size + self.encoder.patch_size // 2).tolist()
        idxs_mask_set_y = (points_matched_inds_set_h * self.encoder.patch_size + self.encoder.patch_size // 2).tolist()

        ponits_matched = []
        for x, y in zip(idxs_mask_set_x, idxs_mask_set_y):
            if int(x) < self.input_size[1] and int(y) < self.input_size[0]:
                ponits_matched.append([int(x), int(y)])
        ponits = np.array(ponits_matched)

        if self.use_box:
            box = np.array([
                max(ponits[:, 0].min(), 0),
                max(ponits[:, 1].min(), 0),
                min(ponits[:, 0].max(), self.input_size[1] - 1),
                min(ponits[:, 1].max(), self.input_size[0] - 1),
            ])
        else:
            box = None

        return ponits, box, S, C, reduced_points_num

    def clustering(self, points):

        num_centers = min(self.num_centers, len(points))
        flag = True
        while (flag):
            centers, cluster_assignment = kmeans_pp(points, num_centers)
            id, fre = torch.unique(cluster_assignment, return_counts=True)
            if id.shape[0] == num_centers:
                flag = False
            else:
                print('Kmeans++ failed, re-run')
        centers = np.array(centers).astype(np.int64)
        return centers


    def mask_generation(self, tar_img_np, points, box, all_ponits, ref_masks_pool, C):
        samples_list, label_list = self.rps.sample_points(points)
        samples_list = list(samples_list[0])
        tar_masks_ori = []
        for point in samples_list:
            point = point / self.input_size[0]
            point = list(point)
            point[0] = list(point[0])
            pred_masks, ious = self.generator.predict(self.tar_img_np, self.tar_images, point)
            pred_masks = F.interpolate(pred_masks[None, ...], self.input_size, mode="bilinear", align_corners=False)
            tar_masks_ori.append(pred_masks[0])
        tar_masks = torch.cat(tar_masks_ori, dim=0)[:,None,...].cpu().numpy() > 0

        # append to original results
        purity = torch.zeros(tar_masks.shape[0])
        coverage = torch.zeros(tar_masks.shape[0])
        emd = torch.zeros(tar_masks.shape[0])

        samples = samples_list[-1]
        labels = torch.ones(tar_masks.shape[0], samples.shape[1])
        samples = torch.ones(tar_masks.shape[0], samples.shape[1], 2)

        # compute scores for each mask
        for i in range(len(tar_masks)):
            purity_, coverage_, emd_, sample_, label_, mask_ = \
                self.rps.get_mask_scores(
                    points=points,
                    masks=tar_masks[i],
                    all_points=all_ponits,
                    emd_cost=C,
                    ref_masks_pool=ref_masks_pool
                )
            assert np.all(mask_ == tar_masks[i])
            purity[i] = purity_
            coverage[i] = coverage_
            emd[i] = emd_

        pred_masks = tar_masks.squeeze(1)
        metric_preds = {
            "purity": purity,
            "coverage": coverage,
            "emd": emd
        }

        scores = self.alpha * emd + self.beta * purity * coverage ** self.exp

        def check_pred_mask(pred_masks):
            if len(pred_masks.shape) < 3:  # avoid only one mask
                pred_masks = pred_masks[None, ...]
            return pred_masks

        pred_masks = check_pred_mask(pred_masks)

        # filter the false-positive mask fragments by using the proposed metrics
        for metric in ["coverage", "emd", "purity"]:
            if self.score_filter_cfg[metric] > 0:
                thres = min(self.score_filter_cfg[metric], metric_preds[metric].max())
                idx = torch.where(metric_preds[metric] >= thres)[0]
                scores = scores[idx]
                samples = samples[idx]
                labels = labels[idx]
                pred_masks = check_pred_mask(pred_masks[idx])

                for key in metric_preds.keys():
                    metric_preds[key] = metric_preds[key][idx]

        #  score-based masks selection, masks merging
        if self.score_filter_cfg["score_filter"]:

            distances = 1 - scores
            distances, rank = torch.sort(distances, descending=False)
            distances_norm = distances - distances.min()
            distances_norm = distances_norm / (distances.max() + 1e-6)
            filer_dis = distances < self.score_filter_cfg["score"]
            filer_dis[..., 0] = True
            filer_dis_norm = distances_norm < self.score_filter_cfg["score_norm"]
            filer_dis = filer_dis * filer_dis_norm

            pred_masks = check_pred_mask(pred_masks)
            masks = pred_masks[rank[filer_dis][:self.num_merging_mask]]
            masks = check_pred_mask(masks)
            masks = masks.sum(0) > 0
            masks = masks[None, ...]

        else:

            topk = min(self.num_merging_mask, scores.size(0))
            topk_idx = scores.topk(topk)[1]
            topk_samples = samples[topk_idx].cpu().numpy()
            topk_scores = scores[topk_idx].cpu().numpy()
            topk_pred_masks = pred_masks[topk_idx]
            topk_pred_masks = check_pred_mask(topk_pred_masks)

            if self.score_filter_cfg["topk_scores_threshold"] > 0:
                # map scores to 0-1
                topk_scores = topk_scores / (topk_scores.max())

            idx = topk_scores > self.score_filter_cfg["topk_scores_threshold"]
            topk_samples = topk_samples[idx]

            topk_pred_masks = check_pred_mask(topk_pred_masks)
            topk_pred_masks = topk_pred_masks[idx]
            mask_list = []
            for i in range(len(topk_samples)):
                mask = topk_pred_masks[i][None, ...]
                mask_list.append(mask)
            masks = np.sum(mask_list, axis=0) > 0
            masks = check_pred_mask(masks)

        return torch.tensor(masks, device=self.device, dtype=torch.float)


    def clear(self):

        self.tar_img = None
        self.tar_images = None
        self.tar_img_np = None

        self.ref_imgs = None
        self.ref_masks_pool = None
        self.nshot = None

        self.encoder_img_size = None
        self.encoder_feat_size = None



class RobustPromptSampler:

    def __init__(
        self,
        encoder_feat_size,
        sample_range,
        max_iterations
    ):
        self.encoder_feat_size = encoder_feat_size
        self.sample_range = sample_range
        self.max_iterations = max_iterations


    def get_mask_scores(self, points, masks, all_points, emd_cost, ref_masks_pool):

        def is_in_mask(point, mask):
            # input: point: n*2, mask: h*w
            # output: n*1
            h, w = mask.shape
            point = point.astype(np.int)
            point = point[:, ::-1]  # y,x
            point = np.clip(point, 0, [h - 1, w - 1])
            return mask[point[:, 0], point[:, 1]]

        ori_masks = masks
        masks = cv2.resize(
            masks[0].astype(np.float32),
            (self.encoder_feat_size, self.encoder_feat_size),
            interpolation=cv2.INTER_AREA)
        if masks.max() <= 0:
            thres = masks.max() - 1e-6
        else:
            thres = 0
        masks = masks > thres

        # 1. emd
        emd_cost_pool = emd_cost[ref_masks_pool.flatten().bool(), :][:, masks.flatten()]
        emd = ot.emd2(a=[1. / emd_cost_pool.shape[0] for i in range(emd_cost_pool.shape[0])],
                      b=[1. / emd_cost_pool.shape[1] for i in range(emd_cost_pool.shape[1])],
                      M=emd_cost_pool.cpu().numpy())
        emd_score = 1 - emd

        labels = np.ones((points.shape[0],))

        # 2. purity and coverage
        assert all_points is not None
        points_in_mask = is_in_mask(all_points, ori_masks[0])
        points_in_mask = all_points[points_in_mask]
        # here we define two metrics for local matching , purity and coverage
        # purity: points_in/mask_area, the higher means the denser points in mask
        # coverage: points_in / all_points, the higher means the mask is more complete
        mask_area = max(float(masks.sum()), 1.0)
        purity = points_in_mask.shape[0] / mask_area
        coverage = points_in_mask.shape[0] / all_points.shape[0]
        purity = torch.tensor([purity]) + 1e-6
        coverage = torch.tensor([coverage]) + 1e-6
        return purity, coverage, emd_score, points, labels, ori_masks

    def combinations(self, n, k):
        if k > n:
            return []
        if k == 0:
            return [[]]
        if k == n:
            return [[i for i in range(n)]]
        res = []
        for i in range(n):
            for j in self.combinations(i, k - 1):
                res.append(j + [i])
        return res

    def sample_points(self, points):
        # return list of arrary

        sample_list = []
        label_list = []
        for i in range(min(self.sample_range[0], len(points)), min(self.sample_range[1], len(points)) + 1):
            if len(points) > 8:
                index = [random.sample(range(len(points)), i) for j in range(self.max_iterations)]
                sample = np.take(points, index, axis=0)  # (max_iterations * i) * 2
            else:
                index = self.combinations(len(points), i)
                sample = np.take(points, index, axis=0)  # i * n * 2

            # generate label  max_iterations * i
            label = np.ones((sample.shape[0], i))
            sample_list.append(sample)
            label_list.append(label)
        return sample_list, label_list



def build_matcher_oss(args):

    # DINOv2, Image Encoder
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
    dinov2 = vits.__dict__[args.dinov2_size](**dinov2_kwargs)

    dinov2_utils.load_pretrained_weights(dinov2, args.dinov2_weights, "teacher")
    dinov2.eval()
    dinov2.to(device=args.device)

    # SAM
    generator = SemanticSAMPredictor(
        build_semantic_sam(
            model_type='T',
            ckpt=args.semantic_sam_weights
        )
    )  # model_type: 'L' / 'T', depends on your checkpint

    score_filter_cfg = {
        "emd": args.emd_filter,
        "purity": args.purity_filter,
        "coverage": args.coverage_filter,
        "score_filter": args.use_score_filter,
        "score": args.deep_score_filter,
        "score_norm": args.deep_score_norm_filter,
        "topk_scores_threshold": args.topk_scores_threshold
    }

    return Matcher(
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
