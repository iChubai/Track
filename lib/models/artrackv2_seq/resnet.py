import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from timm.models.layers import trunc_normal_
from lib.models.layers.patch_embed import PatchEmbed
from lib.models.artrackv2_seq.utils import combine_tokens, recover_tokens

class BaseBackbone(nn.Module):
    def __init__(self, backbone_type='resnet50', pretrained=True, embed_dim=256, seq_len=10):
        super().__init__()
        
        # 根据 backbone_type 选择不同的 ResNet 模型
        if backbone_type == 'resnet18':
            self.backbone = resnet18(pretrained=pretrained)
        elif backbone_type == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # 移除原始 ResNet 的分类头
        self.backbone.fc = nn.Identity()
        
        # 自定义的卷积层用于调整特征维度
        self.conv_adjust = nn.Conv2d(self.backbone.fc.in_features, embed_dim, kernel_size=1)
        # 初始化卷积层权重
        nn.init.kaiming_normal_(self.conv_adjust.weight, mode='fan_out', nonlinearity='relu')
        self.conv_adjust.bias.data.zero_()
        
        # 嵌入层用于轨迹token和命令token
        self.word_embeddings = nn.Embedding(seq_len, embed_dim)
        self.position_embeddings = nn.Embedding(seq_len, embed_dim)
        self.prev_position_embeddings = nn.Embedding(seq_len, embed_dim)
        
        # 输出层偏置
        self.output_bias = nn.Parameter(torch.zeros(embed_dim))
        
        # 将 position_embeddings 初始化为较小的值
        trunc_normal_(self.position_embeddings.weight, std=.02)
        trunc_normal_(self.prev_position_embeddings.weight, std=.02)
        
        # 可选：添加额外的层或模块
        self.return_inter = False
        self.return_stage = ['layer1', 'layer2', 'layer3', 'layer4']
        
        # 添加Dropout层
        self.pos_drop = nn.Dropout(p=0.1)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def generate_square_subsequent_mask(self, sz, sx, ss):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        sum = sz + sx + ss
        mask = (torch.triu(torch.ones(sum, sum)) == 1).transpose(0, 1)
        mask[:, :] = float('-inf')
        mask[:int(sz/2), :int(sz/2)] = 0 # template self
        mask[int(sz/2):sz, int(sz/2):sz] = 0 # dt self
        mask[int(sz/2):sz, sz:sz+sx] = 0 # dt search
        mask[int(sz / 2):sz, -1] = 0  # dt search
        mask[sz:sz+sx, :sz+sx] = 0 # sr dt-t-sr
        mask[sz+sx:, :] = 0 # co dt-t-sr-co
        return mask

    def forward_features(self, z, x, seqs_input):
        B, C, H, W = x.shape
        
        # 对模板图像和搜索区域图像分别进行特征提取
        z_features = self._forward_impl(z)
        x_features = self._forward_impl(x)
        
        # 如果需要返回中间特征
        if self.return_inter:
            return z_features, x_features
        
        # 调整特征维度
        z_adjusted = self.conv_adjust(z_features['layer4'])
        x_adjusted = self.conv_adjust(x_features['layer4'])
        
        # 展平特征
        z_flat = z_adjusted.flatten(2).transpose(1, 2)  # [B, L_z, embed_dim]
        x_flat = x_adjusted.flatten(2).transpose(1, 2)  # [B, L_x, embed_dim]
        
        len_z = z_flat.shape[1]
        len_x = x_flat.shape[1]
        
        # 生成轨迹token和命令token的嵌入
        seqs_input_ = seqs_input.to(torch.int64)
        tgt = self.word_embeddings(seqs_input_).permute(1, 0, 2)  # [seq_len, B, embed_dim]
        
        # 生成位置嵌入
        query_command_embed_ = self.position_embeddings.weight.unsqueeze(1)  # [seq_len, 1, embed_dim]
        prev_embed_ = self.prev_position_embeddings.weight.unsqueeze(1)  # [seq_len, 1, embed_dim]
        query_seq_embed = torch.cat([prev_embed_, query_command_embed_], dim=0)  # [2*seq_len, 1, embed_dim]
        
        query_seq_embed = query_seq_embed.repeat(1, B, 1)  # [2*seq_len, B, embed_dim]
        
        # 添加位置嵌入到序列输入
        tgt += query_seq_embed[:, :tgt.shape[0]]
        
        # 生成掩码
        len_seq = tgt.shape[0]
        mask = self.generate_square_subsequent_mask(len_z, len_x, len_seq).to(tgt.device)  # [L_z+L_x+seq_len, L_z+L_x+seq_len]
        
        # 合并模板、搜索区域和序列特征
        zx = combine_tokens(z_flat, x_flat, mode='direct')  # [B, L_z+L_x, embed_dim]
        zxs = torch.cat((zx, tgt), dim=0)  # [L_z+L_x+seq_len, B, embed_dim]
        
        # 添加Dropout
        zxs = self.pos_drop(zxs)
        
        # 处理序列特征（这里假设有一个Transformer的编码器）
        zxs = self.transformer_encoder(zxs, src_key_padding_mask=None, mask=mask)
        
        # 提取特征
        z_0_feat = zxs[:len_z, :, :]  # [L_z, B, embed_dim]
        z_1_feat = zxs[:len_z, :, :]  # [L_z, B, embed_dim]
        x_feat = zxs[len_z:len_z+len_x, :, :]  # [L_x, B, embed_dim]
        score_feat = zxs[-1, :, :]  # [B, embed_dim]
        
        # 计算可能性
        share_weight = self.word_embeddings.weight.T
        possibility = torch.matmul(x_feat, share_weight)  # [L_x, B, embed_dim] * [embed_dim, seq_len] -> [L_x, B, seq_len]
        out = possibility + self.output_bias.unsqueeze(0).unsqueeze(0)  # [L_x, B, seq_len]
        temp = out.permute(1, 0, 2)  # [B, L_x, seq_len]

        # 取topk
        value, extra_seq = out.topk(dim=-1, k=1)[0], out.topk(dim=-1, k=1)[1]
        for i in range(out.shape[0]):
            value_i, extra_seq_i = out[i, :, :].topk(dim=-1, k=1)[0], out[i, :, :].topk(dim=-1, k=1)[1]
            if i == 0:
                seqs_output = extra_seq_i
                values = value_i
            else:
                seqs_output = torch.cat([seqs_output, extra_seq_i], dim=-1)
                values = torch.cat([values, value_i], dim=-1)

        output = {
            'seqs': seqs_output.permute(1, 0, 2),  # [B, L_x, 1]
            'class': values.permute(1, 0, 2),     # [B, L_x, 1]
            'feat': temp.permute(1, 0, 2),        # [B, L_x, embed_dim]
            "state": "val/test",
            "x_feat": x_adjusted.detach().permute(0, 2, 3, 1),  # [B, H_x, W_x, embed_dim]
            "seq_feat": x_feat.permute(1, 0, 2)                # [B, L_x, embed_dim]
        }

        return output, z_0_feat, z_1_feat, x_feat, score_feat

    def _forward_impl(self, x):
        # 保存每个阶段的特征
        features = {}
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        features['layer1'] = x
        x = self.backbone.layer2(x)
        features['layer2'] = x
        x = self.backbone.layer3(x)
        features['layer3'] = x
        x = self.backbone.layer4(x)
        features['layer4'] = x

        return features

    def forward(self, z, x, seqs_input):
        """
        联合特征提取和关系建模用于基本的ResNet骨干网络。
        Args:
            z (torch.Tensor): 模板图像特征, [B, C, H_z, W_z]
            x (torch.Tensor): 搜索区域图像特征, [B, C, H_x, W_x]
            seqs_input (torch.Tensor): 序列输入, [B, seq_len]

        Returns:
            output (dict): 包含序列输出、分类值、特征等信息
            z_0_feat (torch.Tensor): 调整后的模板图像特征
            z_1_feat (torch.Tensor): 调整后的搜索区域图像特征
            x_feat (torch.Tensor): 序列特征
            score_feat (torch.Tensor): 分数特征
        """
        output, z_0_feat, z_1_feat, x_feat, score_feat = self.forward_features(z, x, seqs_input)
        return output, z_0_feat, z_1_feat, x_feat, score_feat

# 示例用法
if __name__ == "__main__":
    # 创建基础骨干网络实例，使用 ResNet18
    backbone18 = BaseBackbone(backbone_type='resnet18', pretrained=True, embed_dim=256, seq_len=10)
    z = torch.randn(1, 3, 224, 224)
    x = torch.randn(1, 3, 224, 224)
    seqs_input = torch.randint(0, 10, (1, 10))  # 随机生成序列输入
    output18, z_feat18, x_feat18, seq_feat18, score_feat18 = backbone18(z, x, seqs_input)
    print("ResNet18 Features:")
    print(output18)
    print(z_feat18.shape)  # 输出: torch.Size([L_z, 1, 256])
    print(x_feat18.shape)  # 输出: torch.Size([L_x, 1, 256])
    print(seq_feat18.shape)  # 输出: torch.Size([B, L_x, embed_dim])
    print(score_feat18.shape)  # 输出: torch.Size([B, embed_dim])

    # 创建基础骨干网络实例，使用 ResNet50
    backbone50 = BaseBackbone(backbone_type='resnet50', pretrained=True, embed_dim=256, seq_len=10)
    z_feat50, x_feat50, seq_feat50, score_feat50 = backbone50(z, x, seqs_input)
    print("ResNet50 Features:")
    print(output50)
    print(z_feat50.shape)  # 输出: torch.Size([L_z, 1, 256])
    print(x_feat50.shape)  # 输出: torch.Size([L_x, 1, 256])
    print(seq_feat50.shape)  # 输出: torch.Size([B, L_x, embed_dim])
    print(score_feat50.shape)  # 输出: torch.Size([B, embed_dim])
