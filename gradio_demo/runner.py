from __future__ import annotations

import datetime
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

import gradio as gr
import slugify
import torch
import numpy as np
import huggingface_hub
from huggingface_hub import HfApi
from omegaconf import OmegaConf

from segment_anything import sam_model_registry
from dinov2.models import vision_transformer as vits
import dinov2.utils.utils as dinov2_utils

from gradio_demo.oss_ops_inference import main_oss_ops


ORIGINAL_SPACE_ID = ''
SPACE_ID = os.getenv('SPACE_ID', ORIGINAL_SPACE_ID)


class Runner:
    def __init__(self, hf_token: str | None = None):
        self.hf_token = hf_token

        # self.checkpoint_dir = pathlib.Path('checkpoints')
        # self.checkpoint_dir.mkdir(exist_ok=True)

        # oss, ops
        self.prompt_res_g = None
        self.prompt_mask_g = None
        self.tar1_res_g = None
        self.tar2_res_g = None
        self.version = 1

        self.pred_masks = None
        self.pred_mask_lists = None

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        sam_checkpoint = "models/sam_vit_h_4b8939.pth"
        model_type = "default"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)

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
        dinov2 = vits.__dict__["vit_large"](**dinov2_kwargs)

        dinov2_utils.load_pretrained_weights(dinov2, "models/dinov2_vitl14_pretrain.pth", "teacher")
        dinov2.eval()
        dinov2.to(device=device)
        self.dinov2 = dinov2

    def inference_oss_ops(self, prompt, target1, target2, version):

        if version == 'version 1 (🔺 multiple instances  🔻 whole, 🔻 part)':
            self.prompt_res_g, self.tar1_res_g, self.tar2_res_g = prompt['image'], target1, target2
            self.prompt_mask_g = (prompt['mask'][..., 0] != 0)[None, ...] # 1, H, w
            self.version = 1
        elif version == 'version 2 (🔻 multiple instances  🔺 whole, 🔻 part)':
            self.prompt_res_g, self.tar1_res_g, self.tar2_res_g = prompt['image'], target1, target2
            self.prompt_mask_g = (prompt['mask'][..., 0] != 0)[None, ...]  # 1, H, w
            self.version = 2
        else:
            self.prompt_res_g, self.tar1_res_g, self.tar2_res_g = prompt['image'], target1, target2
            self.prompt_mask_g = (prompt['mask'][..., 0] != 0)[None, ...]  # 1, H, w
            self.version = 3

        self.pred_masks, self.pred_mask_lists = main_oss_ops(
            sam=self.sam,
            dinov2=self.dinov2,
            support_img=self.prompt_res_g,
            support_mask=self.prompt_mask_g,
            query_img_1=self.tar1_res_g,
            query_img_2=self.tar2_res_g,
            version=self.version
        )

        text = "Process Successful!"

        return text


    def clear_fn(self):

        self.prompt_res_g, self.tar1_res_g, self.tar2_res_g, self.prompt_mask_g = None, None, None, None
        self.version = 1
        self.pred_masks = None
        self.pred_mask_lists = None

        return [None] * 7


    def controllable_mask_output(self, k):

        color = np.array([30, 144, 255])

        if self.version != 1:

            prompt_mask_res, tar1_mask_res, tar2_mask_res = self.pred_masks

            h, w = prompt_mask_res.shape[-2:]
            prompt_mask_img = prompt_mask_res.reshape(h, w, 1) * color.reshape(1, 1, -1)
            prompt_mask_res = self.prompt_res_g * 0.5 + prompt_mask_img * 0.5

            h, w = tar1_mask_res.shape[-2:]
            tar1_mask_img = tar1_mask_res.reshape(h, w, 1) * color.reshape(1, 1, -1)
            tar1_mask_res = self.tar1_res_g * 0.5 + tar1_mask_img * 0.5

            h, w = tar2_mask_res.shape[-2:]
            tar2_mask_img = tar2_mask_res.reshape(h, w, 1) * color.reshape(1, 1, -1)
            tar2_mask_res = self.tar2_res_g * 0.5 + tar2_mask_img * 0.5

        else:
            prompt_mask_res = self.pred_masks[0]
            tar1_mask_res, tar2_mask_res = self.pred_mask_lists[1:]

            tar1_mask_res = tar1_mask_res[:min(k, len(tar1_mask_res))].sum(0)>0
            tar2_mask_res = tar2_mask_res[:min(k, len(tar2_mask_res))].sum(0) > 0

            h, w = prompt_mask_res.shape[-2:]
            prompt_mask_img = prompt_mask_res.reshape(h, w, 1) * color.reshape(1, 1, -1)
            prompt_mask_res = self.prompt_res_g * 0.5 + prompt_mask_img * 0.5

            h, w = tar1_mask_res.shape[-2:]
            tar1_mask_img = tar1_mask_res.reshape(h, w, 1) * color.reshape(1, 1, -1)
            tar1_mask_res = self.tar1_res_g * 0.5 + tar1_mask_img * 0.5

            h, w = tar2_mask_res.shape[-2:]
            tar2_mask_img = tar2_mask_res.reshape(h, w, 1) * color.reshape(1, 1, -1)
            tar2_mask_res = self.tar2_res_g * 0.5 + tar2_mask_img * 0.5

        return prompt_mask_res/255, tar1_mask_res/255, tar2_mask_res/255
    

    def inference_vos(self, prompt_vid, vid):

        raise NotImplementedError

