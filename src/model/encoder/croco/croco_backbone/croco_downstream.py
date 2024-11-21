# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# CroCo model for downstream tasks
# --------------------------------------------------------
import torch
from einops import rearrange

from .croco import CroCoNet
from .head_downstream import create_dpt_head, create_gauss_head
from ...backbone.backbone_multiview import BackboneMultiview
from .vit_fpn import ViTFeaturePyramid

def croco_args_from_ckpt(ckpt):
    if 'croco_kwargs' in ckpt: # CroCo v2 released models
        return ckpt['croco_kwargs']
    elif 'args' in ckpt and hasattr(ckpt['args'], 'model'): # pretrained using the official code release
        s = ckpt['args'].model # eg "CroCoNet(enc_embed_dim=1024, enc_num_heads=16, enc_depth=24)"
        assert s.startswith('CroCoNet(')
        return eval('dict'+s[len('CroCoNet'):]) # transform it into the string of a dictionary and evaluate it
    else: # CroCo v1 released models
        return dict()
        
        
class CroCoDownstreamBinocular(CroCoNet):

    def __init__(self,
                 **kwargs):
        """ Build network for binocular downstream task
        It takes an extra argument head, that is called with the features 
          and a dictionary img_info containing 'width' and 'height' keys
        The head is setup with the croconet arguments in this init function
        """
        super(CroCoDownstreamBinocular, self).__init__(**kwargs)

        self.backbone = BackboneMultiview(
            feature_channels=64,    #48
            downscale_factor=4,
            no_cross_attn=False,
        )
        self.vit_fpn = ViTFeaturePyramid(
            in_channels=64,     #48
            scale_factors=[1.0, 2.0, 4.0, 8.0],
        )
        self.set_downstream_head()
        self.set_gauss_head()

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        # if not any(k.startswith('dec_blocks2') for k in ckpt):
        #     for key, value in ckpt.items():
        #         if key.startswith('dec_blocks'):
        #             new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)


    def set_downstream_head(self):
        # allocate heads
        self.head = create_dpt_head(self)

    def set_gauss_head(self):
        self.gauss_head = create_gauss_head(self)

    def _set_mask_generator(self, *args, **kwargs):
        """ No mask generator """
        return

    def _set_mask_token(self, *args, **kwargs):
        """ No mask token """
        self.mask_token = None
        return

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head for downstream tasks, define your own head """
        return
        
    def encode_image_pairs(self, img1, img2):
        """ run encoder for a pair of images
            it is actually ~5% faster to concatenate the images along the batch dimension 
             than to encode them separately
        """
        out_list = self._encode_image( torch.cat( (img1,img2), dim=0) )

        return_list = []
        for i in range(len(out_list)):
            out1, out2 = out_list[i].chunk(2, dim=0)
            return_list.append(rearrange(torch.stack((out1, out2), dim=1), "b v ... -> (b v) ..."))

        # out, out2 = out.chunk(2, dim=0)
        # pos, pos2 = pos.chunk(2, dim=0)

        return return_list

    def _encode_image(self, image):
        """
        image has B x 3 x img_size x img_size 
        """
        # embed the image into patches  (x has size B x Npatches x C) 
        # and get position if each return patch (pos has size B x Npatches x 2)
        x, pos = self.patch_embed(image)              
        # add positional embedding without cls token  
        if self.enc_pos_embed is not None:  # used when w/o RoPE2d
            x = x + self.enc_pos_embed[None,...]
        final_output = []
        # apply the transformer blocks and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)
            final_output.append(x)

        final_output[-1] = self.enc_norm(final_output[-1])
        # x = self.enc_norm(x)
        return final_output


    def _decoder(self, f1, pos1, f2, pos2):
        # final_output = [(f1, f2)] # before_projection
        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)
        final_output = [(f1, f2)]

        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            final_output.append((f1, f2))
        # normalize last output
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)
    
    def _gauss_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'gauss_head{head_num}')
        return head(decout, img_shape)

    def forward(self, img1, img2):
        B, C, H, W = img1.size()
        img_shape = (H, W)

        feat_list = self.encode_image_pairs(img1, img2)

        # Multi ViT
        multi_view_feat, _ = self.backbone(torch.stack([img1, img2], dim=1))
        multi_view_feat = rearrange(multi_view_feat, "b v c h w -> (b v) c h w")
        multi_view_feat_vpn = self.vit_fpn(multi_view_feat)

        res = self.head([feat.float() for feat in feat_list], multi_view_feat_vpn, img_shape)
        gauss = self.gauss_head([feat.float() for feat in feat_list], multi_view_feat_vpn, img_shape)

        return res, gauss
    
    def interleave(self, tensor1, tensor2):
        res1 = torch.stack((tensor1, tensor2), dim=1).flatten(0, 1) # (batch, 2, ...) -> (2*batch, ...)
        res2 = torch.stack((tensor2, tensor1), dim=1).flatten(0, 1)
        return res1, res2