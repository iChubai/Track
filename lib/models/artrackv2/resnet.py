import torch
import torch.nn as nn
import torchvision
from timm.models.layers import trunc_normal_
from lib.models.artrackv2.utils import combine_tokens, recover_tokens

def generate_square_subsequent_mask(sz, sx, ss):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    sum = sz + sx + ss
    mask = (torch.triu(torch.ones(sum, sum)) == 1).transpose(0, 1)
    mask[:, :] = 0
    mask[:int(sz/2), :int(sz/2)] = 1 #template self
    mask[int(sz/2):sz, int(sz/2):sz] = 1 # dt self
    mask[int(sz/2):sz, sz:sz+sx] = 1 # dt search
    mask[int(sz / 2):sz, -1] = 1  # dt search
    mask[sz:sz+sx, :sz+sx] = 1 # sr dt-t-sr
    mask[sz+sx:, :] = 1 # co dt-t-sr-co
    return ~mask

class ResNetFactory:
    """
    ResNet模型工厂类，支持灵活切换不同架构
    功能：
        - 统一管理ResNet的构建过程
        - 自动处理不同架构的通道差异
    支持模型：resnet18, resnet34, resnet50, resnet101, resnet152
    """
    @staticmethod
    def build_resnet(name: str, pretrained: bool = True) -> nn.Module:
        """根据名称构建ResNet实例"""
        model_dict = {
            'resnet18': torchvision.models.resnet18,
            'resnet34': torchvision.models.resnet34,
            'resnet50': torchvision.models.resnet50,
            'resnet101': torchvision.models.resnet101,
            'resnet152': torchvision.models.resnet152
        }
        if name not in model_dict:
            raise ValueError(f"Unsupported ResNet type: {name}")
        return model_dict[name](pretrained=pretrained)

class DynamicResNetBackbone(nn.Module):
    """
    动态ResNet主干网络模块
    主要功能：
        1. 支持多种ResNet架构
        2. 自适应特征图尺寸
        3. 自动通道维度对齐
    输入规格：
        - 模板图像: [B,3,H_z,W_z]
        - 搜索区域: [B,3,H_x,W_x]
    输出规格：
        - 模板特征: [B, N_z, D]
        - 搜索特征: [B, N_x, D]
        (N_z = (H_z//stride) * (W_z//stride), D=embed_dim)
    """
    def __init__(self, cfg):
        super().__init__()
        # 从配置获取参数 --------------------------------
        self.resnet_type = cfg.MODEL.BACKBONE.TYPE      # 如 'resnet50'
        self.stride = cfg.MODEL.BACKBONE.STRIDE        # 下采样率 (16或8)
        self.embed_dim = cfg.MODEL.HIDDEN_DIM          # 特征维度 (默认384)
        self.pretrained = True #是否加载预训练权重

        # 初始化ResNet --------------------------------
        self.resnet = ResNetFactory.build_resnet(self.resnet_type, self.pretrained)
        
        # 通道配置 ------------------------------------
        self._setup_channels()  # 根据ResNet类型确定各层通道数
        
        # 网络结构调整 --------------------------------
        self.feature_extractor = self._build_feature_extractor()
        
        # 自适应池化层 --------------------------------
        self.template_pool, self.search_pool = self._build_adaptive_pools(cfg)
        
        # 投影层 --------------------------------------
        self.proj = self._build_projection_layer()
        
        # 位置编码 ------------------------------------
        self.pos_embed_z0,self.pos_embed_z1, self.pos_embed_x = self._init_position_embeddings(cfg)

    def _setup_channels(self):
        """根据ResNet类型配置各层输出通道"""
        # 不同ResNet的stage输出通道对照表
        channel_config = {
            'resnet18': {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512},
            'resnet34': {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512},
            'resnet50': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048},
            'resnet101': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048},
            'resnet152': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        }
        self.channels = channel_config[self.resnet_type]
        # 最终使用的通道数（layer3输出）
        self.out_channels = self.channels['layer3']

    def _build_feature_extractor(self):
        """构建特征提取器（包含指定层）"""
        return nn.Sequential(
            self.resnet.conv1,    # [B,64,H/2,W/2]
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,  # [B,64,H/4,W/4]
            self.resnet.layer1,   # [B,256,H/4,W/4] (resnet50)
            self.resnet.layer2,   # [B,512,H/8,W/8]
            self.resnet.layer3    # [B,1024,H/16,W/16]
        )

    def _build_adaptive_pools(self, cfg):
        """构建自适应池化层"""
        # 计算目标特征图尺寸
        template_size = (cfg.DATA.TEMPLATE.SIZE//self.stride, 
                        cfg.DATA.TEMPLATE.SIZE//self.stride)
        search_size = (cfg.DATA.SEARCH.SIZE//self.stride,
                      cfg.DATA.SEARCH.SIZE//self.stride)
        return (
            nn.AdaptiveAvgPool2d(template_size),  # 模板池化
            nn.AdaptiveAvgPool2d(search_size)     # 搜索池化
        )

# 修改 _build_projection_layer 方法
    def _build_projection_layer(self):
        proj = nn.Sequential(
            nn.Conv2d(self.out_channels, self.embed_dim, kernel_size=1),
            nn.InstanceNorm2d(self.embed_dim)
        )
        nn.init.kaiming_normal_(proj[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(proj[0].bias, 0)
        return proj

    def _init_position_embeddings(self, cfg):
        """初始化可学习的位置编码"""
        # 模板位置编码
        # 添加尺寸校验
        assert cfg.DATA.TEMPLATE.SIZE % self.stride == 0, f"模板尺寸{cfg.DATA.TEMPLATE.SIZE}必须能被stride{self.stride}整除"
        template_h = cfg.DATA.TEMPLATE.SIZE // self.stride
        template_grid = template_h ** 2  
        pos_z0 = nn.Parameter(torch.zeros(1, template_grid, self.embed_dim))
        pos_z1 = nn.Parameter(torch.zeros(1, template_grid, self.embed_dim))
        # 搜索位置编码
        search_grid = (cfg.DATA.SEARCH.SIZE//self.stride)**2
        pos_x = nn.Parameter(torch.zeros(1, search_grid, self.embed_dim))
        # 初始化
        trunc_normal_(pos_z0, std=.02)
        trunc_normal_(pos_z1, std=.02)
        trunc_normal_(pos_x, std=.02)
        return pos_z0,pos_z1, pos_x

    def forward(self, z0, z1, x):
        """
        前向传播核心逻辑
        参数：
            z0: 模板图像 [B,3,H_z,W_z]
            z1: 候选区域图像 [B,3,H_z,W_z]
            x: 搜索图像 [B,3,H_x,W_x]
        返回：
            z0_feat: 模板特征 [B,N_z,D]
            z1_feat: 候选区域特征 [B,N_z,D]
            x_feat: 搜索特征 [B,N_x,D]
        """
        # 模板特征提取 --------------------------------
        z0_feat = self.feature_extractor(z0)        # [B,C,H/16,W/16]
        z0_feat = self.template_pool(z0_feat)       # [B,C,h_z,w_z]
        z0_feat = self.proj(z0_feat)                # [B,D,h_z,w_z]
        z0_feat = z0_feat.flatten(2).permute(0,2,1) # [B,N_z,D]
        z0_feat = z0_feat + self.pos_embed_z0.to(z0_feat.device)  # 显式广播                # 添加位置编码
        
        # 候选区域特征提取 -
        z1_feat = self.feature_extractor(z1)        # [B,C,H/16,W/16]
        z1_feat = self.template_pool(z1_feat)       # [B,C,h_z,w_z]
        z1_feat = self.proj(z1_feat)                # [B,D,h_z,w_z]
        z1_feat = z1_feat.flatten(2).permute(0,2,1) # [B,N_z,D]
        z1_feat = z1_feat + self.pos_embed_z1.to(z1_feat.device)  # 显式广播                # 添加位置编码

        # 搜索特征提取 --------------------------------
        x_feat = self.feature_extractor(x)        # [B,C,H/16,W/16]
        x_feat = self.search_pool(x_feat)         # [B,C,h_x,w_x]
        x_feat = self.proj(x_feat)                # [B,D,h_x,w_x]
        x_feat = x_feat.flatten(2).permute(0,2,1) # [B,N_x,D]
        x_feat = x_feat + self.pos_embed_x.to(x_feat.device)  # 显式广播                # 添加位置编码
        
        return z0_feat,z1_feat, x_feat

    def finetune_track(self, cfg):
        """
        微调适配方法
        功能：
            1. 调整特征图尺寸
            2. 重新初始化位置编码
            3. 冻结指定层
        """
        # 更新配置参数
        self.stride = cfg.MODEL.BACKBONE.STRIDE
        # 重建池化层
        self.template_pool, self.search_pool = self._build_adaptive_pools(cfg)
        # 重新初始化位置编码
        self.pos_embed_z0, self.pos_embed_z1, self.pos_embed_x = self._init_position_embeddings(cfg)
        # 冻结层处理
        if cfg.MODEL.BACKBONE.FREEZE_LAYERS > 0:
            freeze_modules = [
                self.resnet.conv1, self.resnet.bn1,
                self.resnet.layer1, self.resnet.layer2
            ][:cfg.MODEL.BACKBONE.FREEZE_LAYERS]
            for module in freeze_modules:
                for param in module.parameters():
                    param.requires_grad = False

class ARTrackWithResNet(nn.Module):
    """
    ARTrackV2完整模型（ResNet版本）
    主要组件：
        1. ResNetBackbone - 特征提取
        2. Transformer - 特征交互
        3. PredictionHead - 输出预测
    接口与原ViT版本完全兼容
    """
    def __init__(self, cfg):
        super().__init__()
        # 主干网络 --------------------------------
        self.backbone = DynamicResNetBackbone(cfg)
        self.range = cfg.MODEL.RANGE  # 范围限制
        # 轨迹参数 --------------------------------
        self.bins = cfg.MODEL.BINS       # 坐标离散化区间数
        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE  # 特征拼接模式
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG # 是否添加分段标识
        
        # Tokens嵌入层 -----------------------------
        self.word_embeddings = nn.Embedding(self.bins * self.range + 12, cfg.MODEL.HIDDEN_DIM, padding_idx=self.bins * self.range + 4, max_norm=None, norm_type=2.0)
        nn.init.kaiming_normal_(self.word_embeddings.weight.data)
        self.pos_embeddings = nn.Embedding(
            num_embeddings=10,  # 5个坐标位
            embedding_dim=cfg.MODEL.HIDDEN_DIM
        )
        trunc_normal_(self.pos_embeddings.weight, std=.02)
        # Transformer ------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.MODEL.HIDDEN_DIM,
            nhead=cfg.MODEL.NUM_HEADS,
            dim_feedforward=cfg.MODEL.DIM_FEEDFORWARD,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.MODEL.ENCODER_LAYER
        )
        # 预测头 ------------------------------
        self.output = nn.Linear(cfg.MODEL.HIDDEN_DIM, self.bins * self.range + 11)

        self.output_bias = nn.Parameter(torch.zeros(self.bins * self.range + 11))  # 偏置项
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)
        self.norm = nn.LayerNorm(cfg.MODEL.HIDDEN_DIM)
        # 分段标识（可选）---------------------------
        if self.add_sep_seg:
            self.template_segment = nn.Parameter(torch.zeros(1, 1, cfg.MODEL.HIDDEN_DIM))
            self.search_segment = nn.Parameter(torch.zeros(1, 1, cfg.MODEL.HIDDEN_DIM))
            trunc_normal_(self.template_segment, std=.02)
            trunc_normal_(self.search_segment, std=.02)

    def _add_identity(self, features, identity):
        """添加identity连接（与原ViT兼容）"""
        z0_feat, z1_feat, x_feat = features
        z0_feat += identity[:, 0].unsqueeze(1)
        z1_feat += identity[:, 1].unsqueeze(1)
        x_feat += identity[:, 2].unsqueeze(1)
        return z0_feat, z1_feat, x_feat

    def _add_segment_emb(self, features):
        """添加分段标识（可选）"""
        z0_feat, z1_feat, x_feat = features
        z0_feat += self.template_segment
        z1_feat += self.template_segment
        x_feat += self.search_segment
        return z0_feat, z1_feat, x_feat

    def forward_features(self, z_0, z_1, x, identity):
        """
        特征处理流程
        步骤：
            1. 提取双模板特征
            2. 添加identity连接
            3. （可选）添加分段标识
        """
        # 提取特征 --------------------------------
        z0_feat,z1_feat, x_feat = self.backbone(z_0, z_1,x)  # 模板z0与搜索区域
        # 处理特征 --------------------------------
        features = self._add_identity((z0_feat, z1_feat, x_feat), identity)
        if self.add_sep_seg:
            features = self._add_segment_emb(features)
        return features

    def forward_train(self, z_0, z_1, x, identity, seqs_input):
        """训练模式前向传播"""
        # 特征提取 --------------------------------
        z0_feat, z1_feat, x_feat = self.forward_features(z_0, z_1, x, identity)
        share_weight = self.word_embeddings.weight.T

        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        seqs_input = seqs_input.to(torch.int64).to(x.device)
        tgt = self.word_embeddings(seqs_input).permute(1, 0, 2)
        query_embed = self.pos_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, B, 1)

        tgt = tgt.transpose(0, 1)
        query_embed = query_embed.transpose(0, 1)

        len_x = x_feat.shape[1]
        len_z = z0_feat.shape[1] + z1_feat.shape[1]
        len_seq = seqs_input.shape[1]

        mask = generate_square_subsequent_mask(len_z, len_x, len_seq).to(tgt.device)

        tgt += query_embed
        lens_x = x_feat.shape[1]
        z_feat = torch.cat([z0_feat, z1_feat],dim=1)

        x_feat = combine_tokens(z_feat,x_feat,mode = self.cat_mode)

        x_feat = torch.cat([x_feat, tgt], dim=1)


        features = self.transformer(x_feat.transpose(0, 1), mask=mask).transpose(0, 1)

        x_out = self.norm(features[:, -11:-7])
        score_feat = x[:, -7]
        x_n_out = self.norm(features[:, -5:-1])
        score_n_feat = features[:, -1  ]
       
        lens_z = z0_feat.shape[1]
        

        z_0_feat = features[:, :lens_z]
        z_1_feat = features[:, lens_z:lens_z*2]
        x_feat = features[:, lens_z*2:lens_z*2+lens_x]

        #x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)
        at = torch.matmul(x_out, share_weight)
        at = at + self.output_bias
        at = at[:, -4:]
        at = at.transpose(0, 1)

        at_n = torch.matmul(x_n_out, share_weight)
        at_n = at_n + self.output_bias
        at_n = at_n[:, -4:]
        at_n = at_n.transpose(0, 1)

        output = {'feat': at, 'score_feat':score_feat, 'feat_n': at_n,'score_n_feat':score_n_feat, "state": "train"}

        return output, z_0_feat, z_1_feat, x_feat        
    def forward_track(self, z_0, z_1, x, identity):
        """推理模式前向传播（自回归生成）"""
        # # 特征提取 --------------------------------
        # z0_feat, z1_feat, x_feat = self.forward_features(z_0, z_1, x, identity)
        
        # # 初始化Tokens ----------------------------
        # B = x.size(0)
        # device = x.device
        # seq = torch.tensor([
        #     [self.num_bins*0, self.num_bins*1,  # x0, y0
        #      self.num_bins*2, self.num_bins*3,  # x1, y1
        #      self.num_bins*4 + 5]               # score
        # ], device=device).repeat(B, 1)  # [B,5]
        
        # # 自回归生成 ------------------------------
        # for _ in range(5):
        #     # 嵌入当前Tokens
        #     token_emb = self.word_embeddings(seq)  # [B,T,D]
        #     token_emb += self.pos_embeddings(torch.arange(seq.size(1), device=device))
            
        #     # 组合特征
        #     combined = combine_tokens(
        #         torch.cat([z0_feat, z1_feat], dim=1),
        #         x_feat,
        #         token_emb,
        #         mode=self.cat_mode
        #     )
            
        #     # Transformer处理
        #     features = self.transformer(combined.transpose(0,1)).transpose(0,1)
            
        #     # 预测下一个Token
        #     logits = self.output(features[:, -1:]) + self.output_bias
        #     next_token = logits.argmax(dim=-1)
        #     seq = torch.cat([seq, next_token], dim=1)
        
        # return {
        #     'seqs': seq[:, -5:],      # 最终5个Token
        #     'class': logits.softmax(dim=-1),  # 置信度
        #     'feat': features[:, -5:], # 最终特征
        #     'state': 'val/test'
        # }, None, None, None
        share_weight = self.word_embeddings.weight.T
        z_0_feat, z_1_feat, x_feat = self.forward_features(z_0, z_1, x, identity)
        out_list = []

        x0 = self.bins * self.range
        y0 = self.bins * self.range + 1
        x1 = self.bins * self.range + 2
        y1 = self.bins * self.range + 3
        score = self.bins * self.range + 5

        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x0_n = self.bins * self.range + 6
        y0_n = self.bins * self.range + 7
        x1_n = self.bins * self.range + 8
        y1_n = self.bins * self.range + 9
        score_n = self.bins * self.range + 11
        seq = torch.cat([torch.ones((B, 1)).to(x) * x0, torch.ones((B, 1)).to(x) * y0,
                       torch.ones((B, 1)).to(x) * x1,
                       torch.ones((B, 1)).to(x) * y1,
                       torch.ones((B, 1)).to(x) * score,
                       torch.ones((B, 1)).to(x) * x0_n, torch.ones((B, 1)).to(x) * y0_n,
                       torch.ones((B, 1)).to(x) * x1_n,torch.ones((B, 1)).to(x)*y1_n,]
                       [torch.ones((B, 1)).to(x) * score_n], dim=1)

        seq_all = torch.cat([seq], dim=1)

        seqs_input = seq_all.to(torch.int64).to(x.device)
        output_x_feat = x.clone()
        tgt = self.word_embeddings(seqs_input).permute(1, 0, 2)

        len_x = x_feat.shape[1]
        len_z = z_0_feat.shape[1] + z_1_feat.shape[1]
        len_seq = seqs_input.shape[1]

        query_pos_embed = self.pos_embeddings.weight.unsqueeze(1)
        query_pos_embed = query_pos_embed.repeat(1, B, 1)
        
        tgt = tgt.transpose(0, 1)
        query_pos_embed = query_pos_embed.transpose(0, 1)

        
        mask = generate_square_subsequent_mask(len_z, len_x, len_seq).to(tgt.device)

        tgt += query_pos_embed[:, :tgt.shape[1]]

        z_feat = torch.cat([z_0_feat, z_1_feat],dim=1)
        zx_feat = combine_tokens(z_feat, x_feat, mode=self.cat_mode)
        zxs_feat = torch.cat([zx_feat, tgt], dim=1)

        features = self.transformer(zxs_feat.transpose(0, 1), mask=mask).transpose(0, 1)

        lens_z_single = z_0_feat.shape[1]
        lens_x = x_feat.shape[1]
        
        z_0_feat = zxs_feat[:, :lens_z_single]
        z_1_feat = zxs_feat[:, lens_z_single:lens_z_single*2]
        x_feat = zxs_feat[:,  lens_z_single * 2:lens_z_single * 2 + lens_x]

        x_out = self.norm(zxs[:, -11:-7])
        score_feat = x[:, -7]
        
        x_n_out = self.norm(zxs[:, -5:-1])
        score_n_feat = x[:, -1]


        possibility = torch.matmul(x_out, share_weight)
        out = possibility + self.output_bias
        temp = out.transpose(0, 1)
        out_list.append(out.unsqueeze(0))
        out = out.softmax(-1)

        value, extra_seq = out.topk(dim=-1, k=1)[0], out.topk(dim=-1, k=1)[1]
        for i in range(4):
            value, extra_seq = out[:, i, :].topk(dim=-1, k=1)[0], out[:, i, :].topk(dim=-1, k=1)[1]
            if i == 0:
                seqs_output = extra_seq
                values = value
            else:
                seqs_output = torch.cat([seqs_output, extra_seq], dim=-1)
                values = torch.cat([values, value], dim=-1)

        output = {'seqs': seqs_output, 'class': values, 'feat': temp, "state": "val/test",
                  "x_feat": output_x_feat.detach(), "score_feat": score_feat}

        return output, None, None, None

    def forward(self, z_0, z_1, x, identity, seqs_input=None, **kwargs):
        """统一前向接口"""
        if seqs_input is None:
            return self.forward_track(z_0, z_1, x, identity)
        else:
            return self.forward_train(z_0, z_1, x, identity, seqs_input)

    def predict(self, z, x):
        """兼容原项目的预测接口"""
        return self.forward_track(z, z, x, torch.zeros_like(z))[0]['seqs']