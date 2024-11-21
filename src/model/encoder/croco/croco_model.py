from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import nn, Tensor

from .croco_backbone.croco_downstream import CroCoDownstreamBinocular, croco_args_from_ckpt
from .croco_backbone.pos_embed import interpolate_pos_embed
# from .resnet_encoder import ResnetEncoder
# from .posenet import PoseNet

from einops import rearrange, repeat
# from .geometry import pose_inverse_4x4, sample_image_grid, get_world_rays
# from .gaussians import (
#     homogenize_matrices,
#     homogenize_matrices2, 
#     pose_vec2mat, 
#     inverse_warp, 
#     build_covariance,
#     BackprojectDepth,
# )
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as T

@dataclass
class CrocoModelCfg:
    ckpt_path: str
    img_size: list[int]
    adapt: bool


def croco_args_from_ckpt(ckpt):
    if 'croco_kwargs' in ckpt: # CroCo v2 released models
        return ckpt['croco_kwargs']
    elif 'args' in ckpt and hasattr(ckpt['args'], 'model'): # pretrained using the official code release
        s = ckpt['args'].model # eg "CroCoNet(enc_embed_dim=1024, enc_num_heads=16, enc_depth=24)"
        assert s.startswith('CroCoNet(')
        return eval('dict'+s[len('CroCoNet'):]) # transform it into the string of a dictionary and evaluate it
    else: # CroCo v1 released models
        return dict()

class CrocoModel(nn.Module):
    cfg: CrocoModelCfg

    def __init__(self, cfg: CrocoModelCfg) -> None:
        super(CrocoModel, self).__init__()
        self.cfg = cfg
        ckpt = torch.load(cfg.ckpt_path, 'cpu')
        croco_args = croco_args_from_ckpt(ckpt)
        croco_args['img_size'] = tuple(cfg.img_size)
        croco_args['adapt'] = cfg.adapt
        
        self.croco = CroCoDownstreamBinocular(**croco_args)
        interpolate_pos_embed(self.croco, ckpt["model"])
        msg = self.croco.load_state_dict(ckpt["model"], strict=False)
        print(msg)
        for name, p in self.croco.named_parameters():
            if name in msg.missing_keys:
                p.requires_grad = True
            else:
                p.requires_grad = False
        
        del self.croco.decoder_embed
        del self.croco.dec_blocks
        del self.croco.dec_norm
        del self.croco.mask_token

        print()

    def forward(self, img1, img2):

        prediction, gauss = self.croco(img1, img2)    

        return prediction, gauss