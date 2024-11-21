# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# Heads for downstream tasks
# --------------------------------------------------------

"""
A head is a module where the __init__ defines only the head hyperparameters.
A method setup(croconet) takes a CroCoNet and set all layers according to the head and croconet attributes.
The forward takes the features as well as a dictionary img_info containing the keys 'width' and 'height'
"""
from einops import rearrange
from typing import List
import torch
import torch.nn as nn
from .dpt_block import DPTOutputAdapter


class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess
        del self.scratch.layer1_rn
        del self.scratch.layer2_rn
        del self.scratch.layer3_rn
        del self.scratch.layer4_rn

    def forward(self, encoder_tokens: List[torch.Tensor], multi_view_feats: List[torch.Tensor], image_size=None):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        # H, W = input_info['image_size']
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        # layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [torch.cat([l, multi_view_feats[idx]], dim=1) for idx, l in enumerate(layers)]

        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages

        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # Output head
        out = self.head(path_1)

        return out


class PixelwiseTaskWithDPT(nn.Module):
    """ DPT module for CroCo.
    by default, hooks_idx will be equal to:
    * for encoder-only: 4 equally spread layers
    * for encoder+decoder: last encoder + 3 equally spread layers of the decoder 
    """

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, postprocess=nn.Sigmoid(), **kwargs):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_blocks = True # backbone needs to return all layers 
        self.postprocess = postprocess
        assert n_cls_token == 0, 'DPT does not support classification tokens'
        
        dpt_args = dict(
            output_width_ratio=output_width_ratio,
            num_channels=num_channels,
            **kwargs
        )
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)


    def forward(self, x, mv_feat, img_info):
        out = self.dpt(x, mv_feat, image_size=(img_info[0],img_info[1]))
        if self.postprocess: out = self.postprocess(out)
        return out
    

def create_dpt_head(net):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    # assert net.dec_depth == 8
    feature_dim = 256
    last_dim = feature_dim//2
    out_nchan = 1
    enc_dim = net.enc_embed_dim
    return PixelwiseTaskWithDPT(num_channels=out_nchan,
                                feature_dim=feature_dim,
                                last_dim=last_dim,
                                hooks_idx=[6, 11, 17, 23],    # [2, 5, 8, 11]
                                dim_tokens=[enc_dim, enc_dim, enc_dim, enc_dim],
                                postprocess=None,
                                head_type='regression')


def reg_dense_depth(depth, mode):
    """
    extract 3D points from prediction head output
    """
    mode, vmin, vmax = mode

    if mode == 'linear':
        return depth.clip(min=vmin, max=vmax)
    if mode == 'exp':
        # return torch.expm1(depth).clip(min=vmin, max=vmax)
        return vmin + depth.exp().clip(max=vmax-vmin)
    if mode == 'sigmoid':
        disp = depth.sigmoid()
        min_disp = 1.0 / vmax
        max_disp = 1.0 / vmin
        scaled_disp = min_disp + disp * (max_disp - min_disp)
        # scaled_disp = 0.01 + disp * 10.0
        return (1.0 / scaled_disp)
    if mode == 'expm1':
        return vmin + torch.expm1(depth).clip(max=vmax-vmin)
    raise ValueError(f'bad {mode=}')

def reg_dense_conf(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == 'exp':
        return vmin + x.exp().clip(max=vmax-vmin)
    if mode == 'sigmoid':
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f'bad {mode=}')

# @MODIFIED
def reg_dense_offsets(xyz, shift=6.0):
    """
    Apply an activation function to the offsets so that they are small at initialization
    """
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    offsets = xyz * (torch.exp(d - shift) - torch.exp(torch.zeros_like(d) - shift))
    return offsets

# @MODIFIED
def reg_dense_scales(scales):
    """
    Apply an activation function to the offsets so that they are small at initialization
    """
    scales = scales.exp()
    return scales

# @MODIFIED
def reg_dense_rotation(rotations, eps=1e-8):
    """
    Apply PixelSplat's rotation normalization
    """
    return rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

# @MODIFIED
def reg_dense_sh(sh):
    """
    Apply PixelSplat's spherical harmonic postprocessing
    """
    sh = rearrange(sh, '... (xyz d_sh) -> ... xyz d_sh', xyz=3)
    return sh

# @MODIFIED
def reg_dense_opacities(opacities):
    """
    Apply PixelSplat's opacity postprocessing
    """
    return opacities.sigmoid()

def gaussian_postprocess(out, use_offsets=False, sh_degree=1):
    # fmap = out.permute(0, 2, 3, 1)
    # if use_offsets:
    #     offset, scales, rotations, sh, opacities = torch.split(fmap, [2, 3, 4, 3*sh_degree, 1], dim=-1)
    # else:
    #     scales, rotations, sh, opacities = torch.split(fmap, [2, 4, 3*sh_degree, 1], dim=-1)
    #     offset = None

    # # offst = reg_dense_offsets(offset) if use_offsets else None # (B, H, W, 3)
    # # scals = reg_dense_scales(scales) # (B, H, W, 3)
    # # rots = reg_dense_rotation(rotations) # (B, H, W, 4)
    # # shs = reg_dense_sh(sh) # (B, H, W, 3, 1)
    # # opacs = reg_dense_opacities(opacities) # (B, H, W, 1)

    # res = {
    #     'scales': scales,
    #     'rotations': rotations,
    #     'sh': sh,
    #     'opacities': opacities,
    # }

    # if use_offsets:
    #     res['offsets'] = offset
    
    return out

class GaussHead(nn.Module):

    def __init__(self, hooks_idx=None, dim_tokens=None, 
                 num_channels=1, postprocess=None, feature_dim=256, 
                 last_dim=32, head_type="regression", **kwargs):
        super(GaussHead, self).__init__()

        self.gaussian_dpt = PixelwiseTaskWithDPT(
            num_channels=num_channels, feature_dim=feature_dim, last_dim=last_dim,
            hooks_idx=hooks_idx, dim_tokens=dim_tokens, postprocess=postprocess,            
        )

        # final_conv_layer = self.gaussian_dpt.dpt.head[-1]
        # splits_and_inits = [
        #     (3, 0.001, 0.001),  # 3D mean offsets
        #     (3, 0.00003, -7.0),  # Scales
        #     (4, 1.0, 0.0),  # Rotations
        #     (3 * sh_degree, 1.0, 0.0),  # Spherical Harmonics
        #     (1, 1.0, -2.0)  # Opacity
        # ]
        # start_channels = 0
        # for out_channel, s, b in splits_and_inits:
        #     torch.nn.init.xavier_uniform_(
        #         final_conv_layer.weight[start_channels:start_channels+out_channel, :, :, :],
        #         s
        #     )
        #     torch.nn.init.constant_(
        #         final_conv_layer.bias[start_channels:start_channels+out_channel],
        #         b
        #     )
        #     start_channels += out_channel

    def forward(self, x, mv_feats, img_info):
        out = self.gaussian_dpt.dpt(x, mv_feats, img_info)
        return out


def create_gauss_head(net):
    local_feat_dim = 24
    # assert net.dec_depth == 8
    feature_dim = 256
    last_dim = feature_dim//2
    out_nchan=128
    enc_dim = net.enc_embed_dim
    return GaussHead(num_channels=out_nchan,
                     feature_dim=feature_dim,
                     last_dim=last_dim,
                     hooks_idx=[6, 11, 17, 23],    # [2, 5, 8, 11]
                     dim_tokens=[enc_dim, enc_dim, enc_dim, enc_dim], 
                     postprocess=None,
                     head_type='regression',)