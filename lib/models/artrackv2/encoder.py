"""
Encoder modules: we use ITPN for the encoder.
"""

from torch import nn
from lib.utils.misc import is_main_process
from lib.models.artrackv2 import fastitpn as fastitpn_module



class EncoderBase(nn.Module):

    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int):
        super().__init__()
        self.body = encoder
        self.num_channels = num_channels

    def forward(self, z_0, z_1, x, identity, seqs_input, **kwargs):
        xs = self.body(z_0, z_1, x, identity, seqs_input, **kwargs)
        return xs


class Encoder(EncoderBase):
    """FastITPN encoder."""
    def __init__(self, name: str,
                 train_encoder: bool,
                 pretrain_type: str,
                 search_size: int,
                 search_number: int,
                 template_size: int,
                 template_number: int,
                 open_layers: list,
                 cfg=None):
        if "fastitpn" in name.lower():
            encoder = getattr(fastitpn_module, name)(
                pretrained=is_main_process(),
                search_size=search_size,
                template_size=template_size,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                init_values=0.1,
                drop_block_rate=None,
                use_mean_pooling=True,
                grad_ckpt=cfg.MODEL.ENCODER.GRAD_CKPT,
                pos_type=cfg.MODEL.BACKBONE.POS_TYPE,
                pretrain_type = cfg.MODEL.ENCODER.PRETRAIN_TYPE,
            )
            if "itpnb" in name:
                num_channels = 512
            elif "itpnl" in name:
                num_channels = 768
            elif "itpnt" in name:
                num_channels = 384
            elif "itpns" in name:
                num_channels = 384
            else:
                num_channels = 512
        else:
            raise ValueError()
        super().__init__(encoder, train_encoder, open_layers, num_channels)



def build_encoder(cfg):
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    encoder = Encoder(cfg.MODEL.ENCODER.TYPE, train_encoder,
                      cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                      cfg.DATA.SEARCH.SIZE, cfg.DATA.SEARCH.NUMBER,
                      cfg.DATA.TEMPLATE.SIZE, cfg.DATA.TEMPLATE.NUMBER,
                      cfg.TRAIN.ENCODER_OPEN, cfg)
    return encoder