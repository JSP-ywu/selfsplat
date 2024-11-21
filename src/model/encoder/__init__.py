from .encoder import Encoder
from .encoder_self import EncoderSelf, EncoderSelfCfg

ENCODERS = {
    "self": (EncoderSelf),
}

EncoderCfg = EncoderSelfCfg


def get_encoder(cfg: EncoderCfg) -> Encoder:
    encoder = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    return encoder
