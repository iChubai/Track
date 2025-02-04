from copy import deepcopy
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from lib.models.artrackv2.fastitpn import fastitpnt, fastitpns, fastitpnb, fastitpnl
from lib.models.artrackv2.resnet import ARTrackWithResNet
from lib.models.artrackv2.vit import vit_base_patch16_224, vit_large_patch16_224,vit_small_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.artrackv2.refinement_module import build_token_refiner

class ARTrackV2(nn.Module):

    def __init__(self,
                 transformer,
                 refinement_module,
                 score_mlp,
                 ):

        super().__init__()
        self.identity = torch.nn.Parameter(torch.zeros(1, 3, 384))
        self.identity = trunc_normal_(self.identity, std=.02)        
        
        self.backbone = transformer
        self.refinement_module = refinement_module
        self.score_mlp = score_mlp
        
        
    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                seq_input=None,
                target_in_search_img=None,
                gt_bboxes=None,
                ):
        template_0 = template[0]
        template_1 = template[1]
        out, z_0_feat, z_1_feat, x_feat = self.backbone(z_0=template_0, z_1=template_1, x=search, identity=self.identity, seqs_input=seq_input,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn,)
        score_feat = out["score_feat"]
        score = self.score_mlp(score_feat)
        out["score"] = score
        
        # 获取当前帧和下一帧的token
        curr_token = out['feat']  
        next_token = out['feat_n']  
        
        curr_refined,next_refined = self.refinement_module(curr_token, next_token, z_0_feat,z_1_feat,x_feat)
        # 计算当前帧和下一帧的score
        out['feat'] = curr_refined
        
        score_n_feat = out['score_n_feat']
        score_n = self.score_mlp(score_n_feat)
        out['score_n'] = score_n

        return out



class MlpScoreDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, bn=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        out_dim = 1 # score
        if bn:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k), nn.ReLU())
                                          if i < num_layers - 1
                                          else nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])
        else:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.ReLU())
                                          if i < num_layers - 1
                                          else nn.Linear(n, k)
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])

    def forward(self, reg_tokens):
        """
        reg tokens shape: (b, 4, embed_dim)
        """
        x = self.layers(reg_tokens) # (b, 4, 1)
        x = x.mean(dim=1)   # (b, 1)
        return x

def build_score_decoder(cfg, hidden_dim):
    return MlpScoreDecoder(
        in_dim=hidden_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        bn=False
    )


def build_artrackv2(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('ARTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        print("i use vit_large")
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_small_patch16_224':
        print("i use vit_small")
        backbone = vit_small_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'fastitpnt':
        print("i use fastitpnt")
        backbone = fastitpnt(pretrained= True,pretrained_type='pretrained_models/fast_itpn_tiny_1600e_1k.pt', drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION,search_size = cfg.DATA.SEARCH.SIZE,template_size = cfg.DATA.TEMPLATE.SIZE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'fastitpns':
        print("i use fastitpns")
        backbone = fastitpns(pretrained= True,pretrained_type='pretrained_models/fast_itpn_small_1600e_1k.pt', drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'fastitpnb':
        print("i use fastitpnb")
        backbone = fastitpnb(pretrained=True, pretrained_type='pretrained_models/fast_itpn_base_1600e_1k.pt', drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'fastitpnl': 
        print("i use fastitpnl")
        backbone = fastitpnl(pretrained= True,pretrained_type='pretrained_models/fast_itpn_large_1600e_1k.pt', drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE.startswith('resnet'):
        print(f"Using {cfg.MODEL.BACKBONE.TYPE} backbone")
        backbone = ARTrackWithResNet(cfg)
        hidden_dim = cfg.MODEL.HIDDEN_DIM
        patch_start_index = 0 
    else:
        raise NotImplementedError
    if cfg.MODEL.BACKBONE.TYPE.startswith('resnet'):
        backbone.backbone.finetune_track(cfg=cfg)
    else:
         backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    
    refinement_module = build_token_refiner(cfg)
    score_decoder = build_score_decoder(cfg, hidden_dim)

    model = ARTrackV2(
        backbone,        
        refinement_module,
        score_decoder,

    )
    return model
