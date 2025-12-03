import os
from os import path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# import sys
# sys.path.append('/home/yangliu/workspace/pycode/project/SAM/dinov2')

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import torchshow as ts
import cv2
import ot
import math

from matcher_vos.memory import Memory, Frame

from torchvision import transforms

from scipy.optimize import linear_sum_assignment
from matcher_vos.k_means import kmeans_pp
import random
random.seed(0)
from matcher_vos.misc import combinations, get_2d_sincos_pos_embed, show_img_point_box_mask

class Matcher():
    def __init__(
        self,
        encoder,
        predictor,
        input_size=518,
        output_size=None,
        memory_len=0,
        fix_last_frame=True,
        multimask_output=-1,
        pos_embed_ratio=0,
        mask_pool_thres=0.3,
        score_filter=0.8,
        vote_thres=0.5,
        visualize=False,
        vis_path='output',
        point_num_thres_scale=0.05,
        center_dist_filter=2.0,
        memory_decay_ratio=20,
        memory_decay_type='cos'
    ):
        self.encoder = encoder
        self.predictor = predictor        
        
        self.ref_imgs = None
        self.ref_feats = None
        self.ref_masks = None

        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.input_size = input_size
        if output_size and not isinstance(output_size, tuple):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.memory_len = memory_len
        self.memory = Memory(
            memory_len-1,
            fix_last_frame=fix_last_frame,
            memory_decay_ratio=memory_decay_ratio,
            memory_decay_type=memory_decay_type
        )
        self.pos_embed_ratio = pos_embed_ratio
        self.mask_pool_thres = mask_pool_thres
        self.score_filter = score_filter
        self.vote_thres = vote_thres
        self.norm_transform = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std= (0.229, 0.224, 0.225)
        )
            
        self.patch_size = self.encoder.patch_size
        self.mask_generator = MaskGenerator(
            predictor,
            input_size,
            emd_filter=0.75,
            emd_norm_filter=0.85,
            patch_size=self.patch_size,
            multimask_output=multimask_output,
            visualize=visualize,
            vis_path=vis_path,
            point_num_thres_scale=point_num_thres_scale,
            center_dist_filter=center_dist_filter
        )
        
        self.visualize = visualize
        self.vis_path = vis_path
        

        
    def set_reference(self, imgs, masks):
        """
        Args:
            imgs: torch.Tensor in (N,C,H,W) in RGB, values between 0. and 1.
            masks: torch.Tensor in (N,K,H,W) where K is number of objects
        """
        assert imgs.dim() == 4 and masks.dim() == 4
        imgs, imgs_np = self._preprocess_img(imgs)
        masks = self._preprocess_mask(masks)

        ref_feats = self.encoder.forward_features(imgs)['x_prenorm'][:, 1:]
        # ref_feats = ref_feats.reshape(-1, ref_feats.shape[-1])
        self.ref_feats = F.normalize(ref_feats, dim=-1, p=2)
        self.ref_masks = masks
        self.ref_imgs = [imgs_np]
        
        self.memory.clear_memory()
    
    def append_refenrece(self, img, mask):
        """
        Args:
            img: torch.Tensor in (1,C,H,W) in RGB
            mask: torch.Tensor in (1,K,H,W) where K is number of objects
        """
        assert img.dim() == 4 and mask.dim() == 4 and img.shape[0] == 1 and mask.shape[0] == 1
        new_img, new_img_np = self._preprocess_img(img)
        new_mask = self._preprocess_mask(mask)
        
        new_feat = self.encoder.forward_features(new_img)['x_prenorm'][:, 1:]
        # new_feat = new_feat.reshape(-1, new_feat.shape[-1])
        new_feat = F.normalize(new_feat, dim=-1, p=2)
        self.ref_feats = torch.cat((self.ref_feats, new_feat), dim=0)
        
        N, old_K, H, W = self.ref_masks.shape
        new_K = mask.shape[1]
        new_mask = torch.cat((torch.zeros(1, old_K, H, W).to(new_mask.device), new_mask), dim=1)
        self.ref_masks = torch.cat((self.ref_masks, torch.zeros(1, new_K, H, W).to(new_mask.device)), dim=1)
        self.ref_masks = torch.cat((self.ref_masks, new_mask), dim=0)
        self.ref_imgs.append(new_img_np)
        
    def predict(self, tgt):
        # Step 1: Extract feature and compute correspondence matrix
        tgt_img, tgt_frame_id, tgt_name = tgt['tgt_img'], tgt['tgt_frame_id'], tgt['tgt_name']
        tgt_img, tgt_img_np = self._preprocess_img(tgt_img, set_as_target=True)
        tgt_feat = self.encoder.forward_features(tgt_img)['x_prenorm'][:, 1:].squeeze(0)
        tgt_feat = F.normalize(tgt_feat, dim=-1, p=2)
        tgt['tgt_feat'] = tgt_feat
        tgt['tgt_img_np'] = tgt_img_np
        
        # ref_feats = self.ref_feats
        # ref_masks = self.ref_masks
        # DONE memory
        ref = self._assemble_ref()
        ref_frame_ids, ref_imgs, ref_feats, ref_masks, ref_mask_scores = \
            ref['ref_frame_ids'], ref['ref_imgs'], ref['ref_feats'], ref['ref_masks'], ref['ref_mask_scores']
            
        
        n_frame, n_obj, H, W  = ref_masks.shape
        # h, w = H // self.encoder.path_size, W // self.encoder.path_size
        vis_path = path.join(self.vis_path, tgt['tgt_name'])
        os.makedirs(vis_path, exist_ok=True)
        if self.visualize:
        # if tgt['tgt_name'] in ['dogs-jump']:
            for frame_idx in range(n_frame):
                for obj_id in range(n_obj):
                    show_img_point_box_mask(
                        img=ref_imgs[frame_idx],
                        masks=ref_masks[frame_idx, obj_id].cpu().numpy(),
                        mode='mask',
                        save_path=path.join(vis_path, f"{tgt_frame_id:05d}_obj{obj_id}_ref{frame_idx}_{ref_frame_ids[frame_idx]}_score{ref_mask_scores[frame_idx]:.4f}.jpg"),
                    )
        
        
        prompt_points_proposal_list, metric_for_emd_list = \
            self.generate_prompt_proposal(tgt, ref)
        pred_masks = [0.5 * torch.ones((H, W), device=ref_masks.device)]
        mask_scores = []
        for obj_id, (prompt_points_proposal, metric_for_emd) in enumerate(zip(prompt_points_proposal_list, metric_for_emd_list)):
            pred_mask, mask_score = self.mask_generator(tgt, prompt_points_proposal, metric_for_emd, obj_id)
            pred_masks.append(pred_mask * (obj_id + 1))
            mask_scores.append(mask_score)
        pred_masks = torch.stack(pred_masks, dim=0)
        pred_masks_sem = torch.argmax(pred_masks, dim=0).float()
        pred_masks = pred_masks[1:].bool().float()
        mask_scores = sum(mask_scores) / len(mask_scores)
        if self.memory_len:
            self.memory.update_memory(Frame(tgt_img_np, tgt_feat, tgt_frame_id, pred_masks, mask_scores))
        return pred_masks, pred_masks_sem, mask_scores
        

    def generate_prompt_proposal(self, tgt, ref):
        tgt_feat, tgt_frame_id = tgt['tgt_feat'], tgt['tgt_frame_id']
        ref_feats, ref_masks, ref_frame_ids = ref['ref_feats'], ref['ref_masks'], ref['ref_frame_ids']
        ref_masks_pool = F.avg_pool2d(ref_masks, (self.patch_size, self.patch_size))
        ref_masks_pool = (ref_masks_pool > self.mask_pool_thres).float()
        n_ref, n_obj, h, w = ref_masks_pool.shape
        
        pos_metric_list = []
        if self.pos_embed_ratio:
            pos_emb = torch.from_numpy(get_2d_sincos_pos_embed(tgt_feat.shape[-1], h, w)).to(tgt_feat.device)
            pos_metric = pos_emb @ pos_emb.t()
            pos_metric = (pos_metric - pos_metric.min(dim=1).values) / (pos_metric.max(dim=1).values - pos_metric.min(dim=1).values)
            pos_metric_list = []
            for ref_frame_id in ref_frame_ids:
                x = tgt_frame_id - ref_frame_id
                pos_embed_ratio = 0.1 * math.e ** (-x / self.pos_embed_ratio)
                pos_metric_list.append(pos_embed_ratio * pos_metric)
        
        
        metric_for_emd_list = [[] for _ in range(n_obj)]
        point_chosen_times = torch.zeros((n_obj, h*w), device=ref_masks_pool.device)
        for ref_id in range(n_ref):
            obj_mask_start_idx = [0]
            obj_mask_idx = []
            for obj_id in range(n_obj):
                if ref_masks_pool[ref_id, obj_id].sum().item() == 0:
                    ref_masks_pool[ref_id, obj_id] = F.avg_pool2d(ref_masks[ref_id, obj_id][None, ...], (self.patch_size, self.patch_size))
                    ref_masks_pool[ref_id, obj_id] = (ref_masks_pool[ref_id, obj_id] > 0).float()

                obj_mask_start_idx.append(obj_mask_start_idx[-1] + int(ref_masks_pool[ref_id, obj_id].sum().item()))
                obj_mask_idx.append(ref_masks_pool[ref_id, obj_id].flatten().nonzero().squeeze(1))
            obj_mask_idx = torch.cat(obj_mask_idx, dim=0)
            
            ref_feat = ref_feats[ref_id]
            metric = ref_feat @ tgt_feat.t()
            metric_for_emd = (1 - ref_feat @ tgt_feat.t()) / 2
            for obj_id in range(n_obj):
                metric_for_emd_list[obj_id].append(metric_for_emd[ref_masks_pool[ref_id, obj_id].flatten().bool(), :])
            
            # DONE pos embed
            if self.pos_embed_ratio:
                metric = metric + pos_metric_list[ref_id]
                
            
            # bipartite matching
            metric_a = metric[obj_mask_idx]
            indices_a = linear_sum_assignment(metric_a.cpu(), maximize=True)
            indices_a = [torch.as_tensor(index, dtype=torch.int64, device=metric_a.device) for index in indices_a]
            scores_a = metric_a[indices_a[0], indices_a[1]]
            
            # inverse matching
            metric_b = metric.t()[indices_a[1]]
            indices_b = linear_sum_assignment(metric_b.cpu(), maximize=True)
            indices_b = [torch.as_tensor(index, dtype=torch.int64, device=metric_a.device) for index in indices_b]
            

            
            for obj_id in range(n_obj):
                indices_a_0_obj = indices_a[0][obj_mask_start_idx[obj_id]: obj_mask_start_idx[obj_id+1]]
                indices_a_1_obj = indices_a[1][obj_mask_start_idx[obj_id]: obj_mask_start_idx[obj_id+1]]
                indices_b_0_obj = indices_b[0][obj_mask_start_idx[obj_id]: obj_mask_start_idx[obj_id+1]]
                indices_b_1_obj = indices_b[1][obj_mask_start_idx[obj_id]: obj_mask_start_idx[obj_id+1]]
                scores_a_obj = scores_a[obj_mask_start_idx[obj_id]: obj_mask_start_idx[obj_id+1]]
                retain_idx = torch.isin(indices_b_1_obj, obj_mask_idx[obj_mask_start_idx[obj_id]: obj_mask_start_idx[obj_id+1]])
                if (retain_idx==False).all().item():
                    retain_idx = torch.ones_like(retain_idx).bool()
                indices_a_0_obj = indices_a_0_obj[retain_idx]
                indices_a_1_obj = indices_a_1_obj[retain_idx]
                scores_a_obj = scores_a_obj[retain_idx]
                sorted_scores, sorted_indices = torch.sort(scores_a_obj, descending=True)
                if tgt_frame_id < len(self.ref_feats):    # TODO modify for youtube
                    score_filter = len(scores_a_obj)
                else:
                    score_filter = max(10, int(len(scores_a_obj) * self.score_filter))
                indices_a_1_obj = indices_a_1_obj[sorted_indices[:score_filter]]
                point_chosen_times[obj_id][indices_a_1_obj] += 1
        
        point_chosen_times = point_chosen_times / n_ref
        prompt_points_proposal_list = []
        for obj_id in range(n_obj):
            vote_thres = min(self.vote_thres, point_chosen_times[obj_id].max())
            prompt_points_proposal = (point_chosen_times[obj_id] >= vote_thres).nonzero()[:, 0]
            # prompt_points_proposal = torch.tensor(list(set(prompt_points_proposal.cpu().tolist())))
            prompt_points_proposal = torch.stack((
                prompt_points_proposal % w * self.patch_size + self.patch_size // 2,
                prompt_points_proposal // w * self.patch_size + self.patch_size // 2),
                dim=1
            )
            prompt_points_proposal_list.append(prompt_points_proposal)
        
        return prompt_points_proposal_list, metric_for_emd_list
                
        
        
        
    def _preprocess_img(self, imgs, set_as_target=False):
        imgs = F.interpolate(imgs, self.input_size, mode='bicubic', align_corners=False)
        imgs_np = imgs.mul(255).byte().squeeze(0).permute(1,2,0).cpu().numpy()
        if set_as_target:
            self.predictor.set_image(imgs_np)
        imgs = self.norm_transform(imgs)
        return imgs, imgs_np
        
    def _preprocess_mask(self, masks):
        masks = F.interpolate(masks.float(), self.input_size, mode='nearest')
        return masks


    def _assemble_ref(self):
        memory = self.memory.get_memory()
        memory_frame_ids = [frame.frame_id for frame in memory]
        memory_imgs = [frame.img_np for frame in memory]
        memory_feats = [frame.feat[None, ...] for frame in memory]
        memory_masks = [frame.mask[None, ...] for frame in memory]
        memory_mask_scores = [frame.mask_score_decayed for frame in memory]
        
        ref = {
            'ref_frame_ids': [0] *  len(self.ref_feats) + memory_frame_ids,
            'ref_imgs': self.ref_imgs + memory_imgs,
            'ref_feats': torch.cat((self.ref_feats, *memory_feats), dim=0),
            'ref_masks': torch.cat((self.ref_masks, *memory_masks), dim=0),
            'ref_mask_scores': [10.0] * len(self.ref_feats) + memory_mask_scores
        }
        return ref
        
        
        
        
        
        
    
class MaskGenerator():
    def __init__(
        self,
        predictor,
        input_size,
        patch_size,
        center_num=8,
        sample_range=(1, 6),
        max_iteration=300,
        alpha=0.4,
        beta=1.0,
        exp=1,
        purity_filter=0,
        coverage_filter=0,
        emd_filter=0,
        emd_norm_filter=0,
        iou_filter=0,
        topk=5,
        multimask_output=-1,
        point_num_thres_scale=0.05,
        center_dist_filter=2.0,
        visualize=False,
        vis_path='output'
    ):
        self.predictor = predictor
        self.input_size = input_size
        self.patch_size = patch_size
        self.center_num = center_num
        self.sample_range = sample_range
        self.max_iteration = max_iteration
        self.alpha = alpha
        self.beta = beta
        self.exp = exp
        self.purity_filter = purity_filter
        self.coverage_filter = coverage_filter
        self.emd_filter = emd_filter
        self.emd_norm_filter = emd_norm_filter
        self.iou_filter = iou_filter
        self.topk = topk
        self.use_multimask_output = multimask_output != -1
        self.multimask_output_idx = multimask_output
        self.point_num_thres_scale = point_num_thres_scale
        self.center_dist_filter = center_dist_filter
        
        self.visualize = visualize
        self.vis_path = vis_path
    
    def _instance_level_matching(
        self,
        metric_for_emd,
        prompt_points,
        all_points,
        prompt_labels=None,
        prompt_box=None
    ):
        if prompt_labels is None:
            prompt_labels = np.array([1, ] * len(prompt_points))
        prompt_points = prompt_points.cpu().numpy()
        all_points = all_points.cpu().numpy()
        ori_mask, iou_score, low_res_mask = self.predictor.predict(
            point_coords=prompt_points,
            point_labels=prompt_labels,
            box=prompt_box,
            multimask_output=self.use_multimask_output  # TODO
        )
        if self.use_multimask_output:
            ori_mask = ori_mask[self.multimask_output_idx][None, ...]
            iou_score = iou_score[self.multimask_output_idx]
            low_res_mask = low_res_mask[self.multimask_output_idx][None, ...]
        mask = self._convert_mask(low_res_mask)
        emd_score_list = []
        for metric in metric_for_emd:
            metric = metric[:, mask.flatten().bool()].cpu().numpy()
            emd_a = [1 / metric.shape[0] for _ in range(metric.shape[0])]
            emd_b = [1 / metric.shape[1] for _ in range(metric.shape[1])]
            emd_M = metric
            emd_distance = ot.emd2(a=emd_a, b=emd_b, M=emd_M)
            emd_score = 1 - emd_distance
            emd_score_list.append(emd_score)
        emd_score = max(emd_score_list)
        points_in_mask = self._is_in_mask(all_points, ori_mask[0])
        points_in_mask = all_points[points_in_mask]
        mask_area = max(float(mask.sum()), 1)
        purity = points_in_mask.shape[0]/mask_area
        coverage = points_in_mask.shape[0]/all_points.shape[0]
        purity = torch.tensor([purity]) + 1e-6
        coverage = torch.tensor([coverage]) + 1e-6
        return purity, coverage, emd_score, iou_score, ori_mask
        
            
    def _is_in_mask(self, points, mask):
        h, w = mask.shape
        points = points.astype(np.int)
        points = points[:, ::-1] # y,x
        points = np.clip(points, 0, [h-1, w-1])
        return mask[points[:, 0], points[:, 1]]
        
        
        
    def _convert_mask(self, mask):
        mask = torch.from_numpy(mask).to(self.predictor.device)[None, ...]
        mask = self.predictor.model.postprocess_masks(mask, self.predictor.input_size, self.predictor.original_size)
        mask = F.avg_pool2d(mask, (self.patch_size, self.patch_size))
        mask = mask.squeeze()
        thres = self.predictor.model.mask_threshold
        if (mask < 0).all():
            thres = mask.max()
        mask = mask >= thres
        return mask
    
    def _filter_proposals(self, metric, thres, proposals, idx=None):
        if idx is None:
            idx = torch.where(metric >= thres)[0]
        for k in proposals:
            proposals[k] = proposals[k][idx]
            if k == 'masks' and proposals[k].ndim < 3:
                proposals[k] = proposals[k][None, ...]
        return proposals  
    
        
    def __call__(self, tgt, prompt_points_proposal, metric_for_emd, obj_id):
        
        # if tgt['tgt_name'] in ['dogs-jump']:
        #     self.visualize = True
        # else:
        #     self.visualize = False
        vis_path = path.join(self.vis_path, tgt['tgt_name'])
        colors = np.random.rand(self.center_num, 3)
        if self.visualize:
            show_img_point_box_mask(
                img=tgt['tgt_img_np'],
                input_point=prompt_points_proposal.cpu().numpy(),
                input_label=np.array([1, ] * len(prompt_points_proposal)),
                mode='point',
                save_path=path.join(vis_path, f"{tgt['tgt_frame_id']:05d}_obj{obj_id}"+"_point.jpg")
            )
        prompt_points = prompt_points_proposal
        if self.center_num:
            center_num = min(self.center_num, len(prompt_points_proposal))
            while True:
                centers, cluster_assignment = kmeans_pp(prompt_points_proposal.cpu().numpy().astype(np.int64), center_num)
                indices, freq = torch.unique(cluster_assignment, return_counts=True)
                if indices.shape[0] == center_num:
                    break
                else:
                    print("K-means++ fails, re-run")
            
            # TODO point_thres_scale point_thres_bias
            #  filter cluster
            # point_num_thres_bias = -1
            # point_num_thres = max(torch.median(torch.bincount(cluster_assignment)) - point_num_thres_bias,2)
            # point_num_thres = min(point_num_thres, torch.bincount(cluster_assignment).max(), len(prompt_points_proposal) // self.point_num_thres_scale)
            point_num_thres = int(freq.median() * self.point_num_thres_scale)
            # point_num_thres = max(2, int(freq.median() * self.point_num_thres_scale))
            # point_num_thres = min(point_num_thres, freq.median())
            # point_num_thres = min(point_num_thres, freq.max())
            # point_num_thres = min(freq.median(), int(len(prompt_points_proposal) * self.point_num_thres_scale))
            # if self.visualize and center_num > 1:
            #     for k in range(center_num):
            #         # plt.scatter(X[cluster_assignment == k, 0], X[cluster_assignment == k, 1], c=colors[k])
            #         plt.scatter(prompt_points_proposal[(cluster_assignment == k).nonzero()[:,0], 0].cpu(), prompt_points_proposal[(cluster_assignment == k).nonzero()[:,0], 1].cpu(), c=colors[k])
                    
            #     plt.scatter(centers[:, 0].cpu(), centers[:, 1].cpu(), marker='*', s=200, c='black')
            #     # save figure
            #     plt.savefig(path.join(vis_path,f'{tgt["tgt_frame_id"]:05d}_obj{obj_id}_avg_points{len(prompt_points_proposal)}_kmeans.jpg'))
            #     # clear scatter
            #     plt.close()
            
            center_filter = freq >= point_num_thres
            dist_matrix = torch.norm(centers[:, None] - centers, 2, 2)
            dist_sum, dist_indices = dist_matrix.sum(dim=0).sort(descending=True)
            dist_sum_left_shift = torch.cat((dist_sum[1:], dist_sum[-1][None]), dim=0)
            gap_index = (dist_sum > dist_sum_left_shift*self.center_dist_filter).nonzero()
            if len(gap_index):
                center_filter[dist_indices[:gap_index[-1]+1]] = False
            
            
            # dist_filter = dist_sum <= dist_sum_left_shift * 100
            
            
            # TODO fitler_before_ransac or not ?
            prompt_points = centers[center_filter]
            # prompt_points = centers.int() # TODO int or float
            # cluster_assignment2 = cluster_assignment[torch.isin(cluster_assignment, centers_filter_id)]
            if tgt['tgt_name'] in ['dogs-jump']:
                print(f"{tgt['tgt_name']}{tgt['tgt_frame_id']} freq:{freq[dist_indices].cpu().tolist()} thres:{point_num_thres} dist:{dist_sum.int().cpu().tolist()} num: {len(prompt_points)}")
            
            
            if self.visualize and center_num > 1:
                plt.figure(figsize=(10, 10))
                plt.imshow(tgt["tgt_img_np"])  
                for k in range(center_num):
                    plt.scatter(prompt_points_proposal[(cluster_assignment == k).nonzero()[:,0], 0].cpu(), prompt_points_proposal[(cluster_assignment == k).nonzero()[:,0], 1].cpu(), c=colors[k])
                plt.scatter(prompt_points[:, 0].cpu(), prompt_points[:, 1].cpu(), marker='*', s=200, c='red')
                # save figure
                plt.savefig(path.join(vis_path,f'{tgt["tgt_frame_id"]:05d}_obj{obj_id}_avg_points{len(prompt_points_proposal)}_kmeans_filter.jpg'))
                plt.close()
        
        if len(prompt_points) < self.sample_range[0]:
            purity_, coverage_, emd_, iou_, mask_ = \
                self._instance_level_matching(metric_for_emd, prompt_points, prompt_points_proposal)
            purity = torch.tensor(purity_)
            coverage = torch.tensor(coverage_)
            samples = torch.zeros(1, self.sample_range[0], 2)
            samples[:, :len(prompt_points)] = torch.tensor(prompt_points)
            emd = torch.tensor(emd_)
            emd_norm = torch.tensor(1)
            iou = torch.tensor(iou_)
            masks = mask_
        else:
            all_comb = []
            if len(prompt_points) > 15:
                sample_range = self.sample_range[1] - self.sample_range[0] + 1
                all_comb = [random.sample(
                    range(len(prompt_points)),
                    range(self.sample_range[0], self.sample_range[1]+1)[i % sample_range]) \
                    for i in range(self.max_iterations)] 
            else:
                for i in range(self.sample_range[0], min(self.sample_range[1], len(prompt_points)) + 1):
                    all_comb += combinations(len(prompt_points), i)
            purity = torch.zeros(len(all_comb))
            coverage = torch.zeros(len(all_comb))
            emd = torch.zeros(len(all_comb))
            emd_norm = torch.zeros(len(all_comb))
            iou = torch.zeros(len(all_comb))
            samples = torch.zeros(len(all_comb), self.sample_range[1], 2)
            masks = np.zeros((len(all_comb), self.predictor.original_size[0], self.predictor.original_size[1]), dtype=bool)
            for i in range(len(all_comb)):
                samples_ = prompt_points[all_comb[i]]
                purity_, coverage_, emd_, iou_, mask_ = \
                    self._instance_level_matching(metric_for_emd, samples_, prompt_points_proposal)
                assert samples_.shape[0] <= self.sample_range[1]
                purity[i] = purity_
                coverage[i] = coverage_
                samples[i][:samples_.shape[0]] = samples_
                iou[i] = torch.tensor(iou_)
                emd[i] = emd_
                masks[i] = mask_
            emd_norm = (emd - emd.min()) / (emd.max() - emd.min() + 1e-6)
        scores = self.alpha * emd + self.beta * purity * coverage ** self.exp
        proposals = {
            'samples': samples,
            'scores': scores,
            'masks': masks,
            'purity': purity,
            'coverage': coverage,
            'emd': emd,
            'emd_norm': emd_norm,
            'iou': iou
        }
        # TODO filter的顺序
        if self.purity_filter:
            purity_filter = min(self.purity_filter, purity.max())
            proposals = self._filter_proposals(proposals['purity'], purity_filter, proposals)
        if self.coverage_filter:
            coverage_filter = min(self.coverage_filter, coverage.max())
            proposals = self._filter_proposals(proposals['coverage'], coverage_filter, proposals)
        if self.emd_filter:
            emd_filter = min(self.emd_filter, emd.max())
            proposals = self._filter_proposals(proposals['emd'], emd_filter, proposals)
        if self.emd_norm_filter:
            emd_norm_filter = min(self.emd_norm_filter, emd_norm.max())
            proposals = self._filter_proposals(proposals['emd_norm'], emd_norm_filter, proposals)
        if self.iou_filter:
            iou_filter = min(self.iou_filter, iou.max())
            proposals = self._filter_proposals(proposals['iou'], iou_filter, proposals)
        scores, idx_unique = torch.unique(proposals['scores'], return_inverse=True)
        samples = torch.zeros(scores.size(0), proposals['samples'].size(1), proposals['samples'].size(2))
        pred_masks = np.zeros((scores.size(0), self.input_size[0], self.input_size[1]), dtype=bool)
        for i in range(scores.size(0)):
            samples[i] = proposals['samples'][torch.where(idx_unique == i)[0][0]]
            pred_masks[i] = proposals['masks'][torch.where(idx_unique == i)[0][0]]
        topk = min(self.topk, scores.size(0))
        topk_idx = scores.topk(topk)[1]
        # if len(topk_idx) == 0:
        #     raise ValueError()
        topk_samples = samples[topk_idx].cpu().numpy()
        topk_scores = scores[topk_idx].cpu().numpy()
        topk_pred_masks = pred_masks[topk_idx]
        if topk_pred_masks.ndim < 3:
            topk_pred_masks = topk_pred_masks[None, ...]
        
        
        
        # TODO topk_scores_threshold
        topk_scores_threshold = 0.7
        idx = topk_scores >= topk_scores_threshold
        # TODO
        if idx.sum() < max(idx.shape[0] // 2, 1):       
            idx[:max(idx.shape[0] // 2, 1)] = True

        if idx.sum() == 0:
            raise ValueError()
        topk_scores = topk_scores[idx]
        topk_samples = topk_samples[idx]
        topk_pred_masks = topk_pred_masks[idx]
        if topk_pred_masks.ndim < 3:
            topk_pred_masks = topk_pred_masks[None, ...]
        
        # TODO use_box_in_ransac
        
        
        mask_list = []
        score_list = []
        # TODO 并行化
        for i in range(len(topk_samples)):
            mask_list.append(topk_pred_masks[i])
            score_list.append(topk_scores[i])
            if self.visualize:
                valid_node_num = (topk_samples[i].sum(-1) != 0).sum()
                show_img_point_box_mask(
                    img=tgt['tgt_img_np'],
                    masks=topk_pred_masks[i],
                    mode='mask',
                    save_path=path.join(vis_path, f'{tgt["tgt_frame_id"]:05d}_obj{obj_id}_metric{topk_scores[i]:.4f}_mask.jpg'),
                    )
                show_img_point_box_mask(
                    img=tgt['tgt_img_np'],
                    input_point=topk_samples[i][:valid_node_num],
                    input_label=np.array([1, ] * valid_node_num),
                    mode='point',
                    save_path=path.join(vis_path, f'{tgt["tgt_frame_id"]:05d}_obj{obj_id}_metric{topk_scores[i]:.4f}_selectpoints{valid_node_num}.jpg'),
                    )
        mask = np.sum(mask_list, axis=0) > 0
        avg_score = sum(score_list) / len(score_list)
        
        mask = torch.tensor(mask, device=self.predictor.model.device, dtype=torch.float)
        return mask, avg_score
        
        
        
        
        