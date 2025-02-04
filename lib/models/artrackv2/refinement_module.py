
import torch
from torch import nn
import torch.nn.functional as F
import math
class TokenRefiner(nn.Module):  
    def __init__(self, token_dim: int, feat_dim: int):  
        """  
        Args:  
            token_dim: word embedding后的token维度  
            feat_dim: 视觉特征维度  
        """  
        super().__init__()  
        self.token_dim = token_dim  
        self.feat_dim = feat_dim  
        
        # Token特征处理  
        self.token_encoder = nn.Sequential(  
            nn.Linear(token_dim, feat_dim//2),  
            nn.LayerNorm(feat_dim//2),  
            nn.GELU()  
        )  
        
        # 运动特征编码  
        self.motion_encoder = nn.Sequential(  
            nn.Linear(token_dim, feat_dim//2),  
            nn.LayerNorm(feat_dim//2),  
            nn.GELU()  
        )  
        
        # 视觉特征压缩  
        self.feat_compressor = nn.Sequential(  
            nn.Conv1d(feat_dim, feat_dim//2, 1, groups=2),  
            nn.GELU()  
        )  
        
        # Token优化器  
        self.token_refiner = nn.Sequential(  
            nn.Linear(feat_dim, feat_dim//2),  
            nn.LayerNorm(feat_dim//2),  
            nn.GELU(),  
            nn.Linear(feat_dim//2, token_dim)  
        )  
        
        self.register_buffer('prev_prediction', None)  
    
    def process_visual_features(self, z0_feat, z1_feat, x_feat):  
        """处理视觉特征  
        Args:  
            z0_feat: [B,M,D_feat] 第一个模板特征  
            z1_feat: [B,M,D_feat] 第二个模板特征  
            x_feat: [B,M,D_feat] 搜索区域特征  
        Returns:  
            x_comp: [B,M,D_feat//2] 压缩后的搜索区域特征  
            template_feat: [B,M,D_feat//2] 融合后的模板特征  
        """  
        # 压缩特征  
        z0_comp = self.feat_compressor(z0_feat.transpose(-2,-1)).transpose(-2,-1)  
        z1_comp = self.feat_compressor(z1_feat.transpose(-2,-1)).transpose(-2,-1)  
        x_comp = self.feat_compressor(x_feat.transpose(-2,-1)).transpose(-2,-1)  
        
        # 计算模板相似度  
        template_sim = F.cosine_similarity(  
            z0_comp.mean(1),  
            z1_comp.mean(1),  
            dim=-1  
        ).unsqueeze(-1).unsqueeze(-1)  # [B,1,1]  
        
        # 模板特征自适应融合  
        template_feat = template_sim * z0_comp + (1 - template_sim) * z1_comp  
        
        return x_comp, template_feat  
    
    def refine_current_token(self, curr_token, visual_feat, template_feat):  
        """优化当前帧token  
        Args:  
            curr_token: [4,B,D_token] 当前帧token  
            visual_feat: [B,M,D_feat//2] 搜索区域特征  
            template_feat: [B,M,D_feat//2] 模板特征  
        Returns:  
            refined_token: [4,B,D_token] 优化后的token  
        """  
        # 调整token形状以便处理  
        curr_token = curr_token.permute(1,0,2)  # [B,4,D_token]  
        
        # 1. 特征编码  
        curr_feat = self.token_encoder(curr_token)  # [B,4,D_feat//2]  
        
        # 2. 利用上一帧预测结果  
        if self.prev_prediction is not None:  
            prev_token = self.prev_prediction.permute(1,0,2)  # [B,4,D_token]  
            
            # 计算运动特征  
            motion = curr_token - prev_token  # [B,4,D_token]  
            motion_feat = self.motion_encoder(motion)  # [B,4,D_feat//4]  
            
            # 计算预测可信度  
            confidence = F.cosine_similarity(  
                curr_token.view(curr_token.shape[0],4,-1),  
                prev_token.view(prev_token.shape[0],4,-1),  
                dim=-1  
            ).unsqueeze(-1)  # [B,4,1]  
            
            # 特征融合  
            curr_feat = curr_feat + confidence * motion_feat  
        
        # 3. 视觉引导优化  
        feat_sim = torch.bmm(  
            curr_feat,  
            visual_feat.transpose(-2,-1)  
        ) / math.sqrt(curr_feat.shape[-1])  # [B,4,M]  
        
        attn = F.softmax(feat_sim, dim=-1)  
        context_feat = torch.bmm(attn, visual_feat)  # [B,4,D_feat//2]  
        
        # 4. 特征融合与优化  
        fusion_feat = torch.cat([curr_feat, context_feat], dim=-1)  # [B,4,D_feat]  
        delta = self.token_refiner(fusion_feat)  # [B,4,D_token]  
        
        # 5. 生成优化结果  
        if self.prev_prediction is not None:  
            refined = confidence * prev_token + (1 - confidence) * (curr_token + delta)  
        else:  
            refined = curr_token + delta  
            
        # 恢复原始形状  
        refined = refined.permute(1,0,2)  # [4,B,D_token]  
        return refined  
    
    def refine_next_token(self, next_token, curr_refined, visual_feat, template_feat):
        """优化下一帧预测token"""
        
        next_token = next_token.permute(1, 0, 2)  # [B,4,D_token]
        curr_refined = curr_refined.permute(1, 0, 2)  # [B,4,D_token]
        
        # 1. 特征编码
        next_feat = self.token_encoder(next_token)  # [B,4,D_feat//2]
        
        # 2. 计算目标运动特征
        if self.prev_prediction is not None:
            prev_token = self.prev_prediction.permute(1,0,2)  # [B,4,D_token]  
            
            # 基于上一帧的预测（prev_prediction）计算运动
            motion = curr_refined - prev_token  # [B,4,D_token]
            
            # 运动特征编码（例如速度、加速度等）
            motion_feat = self.motion_encoder(motion)  # [B,4,D_feat//4]
            
            # 可调节加权：根据运动情况来更新next_feat
            next_feat = next_feat + motion_feat  # 加入运动信息

        # 3. 视觉引导优化
        feat_sim = torch.bmm(
            next_feat,
            visual_feat.transpose(-2, -1)
        ) / math.sqrt(next_feat.shape[-1])  # [B,4,M]
        
        attn = F.softmax(feat_sim, dim=-1)
        context_feat = torch.bmm(attn, visual_feat)  # [B,4,D_feat//2]
        
        # 4. 特征融合与优化
        fusion_feat = torch.cat([next_feat, context_feat], dim=-1)  # [B,4,D_feat]
        
        # 5. 运动状态自适应调整
        delta = self.token_refiner(fusion_feat)  # [B,4,D_token]

        # 6. 生成优化结果
        refined = next_token + delta  # 基于视觉和运动调整位置
        
        # 恢复原始形状
        refined = refined.permute(1, 0, 2)  # [4,B,D_token]
        
        return refined

    
    def forward(  
        self,  
        curr_token: torch.Tensor,  # [4,B,D_token]  
        next_token: torch.Tensor,  # [4,B,D_token]  
        z0_feat: torch.Tensor,     # [B,M,D_feat]  
        z1_feat: torch.Tensor,     # [B,M,D_feat]  
        x_feat: torch.Tensor,      # [B,M,D_feat]  
        is_first: bool = False  
    ):  
        if is_first:  
            self.reset_state()  
            
        # 1. 处理视觉特征  
        visual_feat, template_feat = self.process_visual_features(z0_feat, z1_feat, x_feat)  
        
        # 2. 优化当前帧token  
        curr_refined = self.refine_current_token(curr_token, visual_feat, template_feat)  
        
        # 3. 优化下一帧预测token  
        next_refined = self.refine_next_token(next_token, curr_refined, visual_feat, template_feat)  
        
        # 4. 更新状态 - 保持[4,B,D_token]形状  
        self.prev_prediction = next_refined.detach()  
        
        return curr_refined, next_refined  
    
    def reset_state(self):  
        """重置状态"""  
        self.prev_prediction = None  

def build_token_refiner(cfg):  
    """构建TokenRefiner"""  
    token_dim = cfg.MODEL.BINS*cfg.MODEL.RANGE+12  # word embedding维度  
    feat_dim = cfg.MODEL.HIDDEN_DIM  # 视觉特征维度  
    return TokenRefiner(token_dim=token_dim, feat_dim=feat_dim)