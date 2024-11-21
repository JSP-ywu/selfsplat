from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from .loss import Loss
from fused_ssim import fused_ssim
import math

@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def __init__(self, cfg: LossMseCfgWrapper) -> None:
        super().__init__(cfg)
        
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        # gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        gt = torch.cat([batch["context"]["image"][:, :1], batch["target"]["image"], batch["context"]["image"][:, -1:]], dim=1)
        delta_ssim = 1.0 - fused_ssim(prediction.color.flatten(0, 1), gt.flatten(0, 1), padding="valid")
        delta = ((prediction.color - gt)**2).mean()
            
        color_rendering_loss = 0.8*delta + 0.2*delta_ssim

        return self.cfg.weight * color_rendering_loss
