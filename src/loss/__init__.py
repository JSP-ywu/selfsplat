from .loss import Loss
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_repro import LossRepro, LossReproCfgWrapper

LOSSES = {
    LossMseCfgWrapper: LossMse,
    LossReproCfgWrapper: LossRepro,
}

LossCfgWrapper = LossMseCfgWrapper | LossReproCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
