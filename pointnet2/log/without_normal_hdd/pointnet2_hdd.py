import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False, transformer_config=None, group_all=False):
        super(get_model, self).__init__()
        if transformer_config is None:
            transformer_config = [0, 0, 0, 0, 0, 0, 0, 0]
        self.insert_positions = transformer_config
        dims = [160, 320, 512, 1024, 256, 256, 128, 128]
        
        self.transformers_dict = nn.ModuleDict()
        for i, num_layers in enumerate(transformer_config):
            if num_layers > 0:
                stage_transformers = nn.ModuleList()
                for _ in range(num_layers):
                    stage_transformers.append(TransformerBlock(dims[i]))
                self.transformers_dict[str(i)] = stage_transformers


        self.sa0 = PointNetSetAbstractionMsg(1024, [0.05, 0.1, 0.2], [16, 32, 64], 3, [[16, 16, 32], [32, 32, 64], [32, 48, 64]])
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 32+64+64, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(16, [0.8, 1.6], [64, 128], in_channel=512+3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 512, mlp=[512, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 320, mlp=[256, 256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256 + 160, mlp=[256, 128])
        # self.fp0 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.fp0 = PointNetFeaturePropagation(in_channel=134, mlp=[128, 128])
        
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

        
    
    def apply_transformer(self, points, stage, dim):
        if str(stage) not in self.transformers_dict:
            return points

        transformers = self.transformers_dict[str(stage)]
        points = points.permute(0, 2, 1)  # [batch_size, num_points, channels]
        # print("points device:", points.device)
        for transformer in transformers:
            points = transformer(points, dim)
        points = points.permute(0, 2, 1)  # revert to [batch_size, channels, num_points]
        return points

    def forward(self, xyz, cls_label=None):
        l0_xyz, l0_points = xyz, xyz
        l1_xyz, l1_points = self.sa0(l0_xyz, l0_points)
        if self.insert_positions[0] > 0:
            l1_points = self.apply_transformer(l1_points, 0, 160)

        l2_xyz, l2_points = self.sa1(l1_xyz, l1_points)
        if self.insert_positions[1] > 0:
            l2_points = self.apply_transformer(l2_points, 1, 320)
        
        l3_xyz, l3_points = self.sa2(l2_xyz, l2_points)
        if self.insert_positions[2] > 0:
            l3_points = self.apply_transformer(l3_points, 2, 512)
        
        l4_xyz, l4_points = self.sa3(l3_xyz, l3_points)
        if self.insert_positions[3] > 0:
            l4_points = self.apply_transformer(l4_points, 3, 1024)
        
        l3_points = self.fp3(l3_xyz, l4_xyz, l3_points, l4_points)
        if self.insert_positions[4] > 0:
            l3_points = self.apply_transformer(l3_points, 4, 256)

        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        if self.insert_positions[5] > 0:
            l2_points = self.apply_transformer(l2_points, 5, 256)
        
        l1_points = self.fp1(l1_xyz, l2_xyz, l1_points, l2_points)
        if self.insert_positions[6] > 0:
            l1_points = self.apply_transformer(l1_points, 6, 128)

        l0_points = self.fp0(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        if self.insert_positions[7] > 0:
            l0_points = self.apply_transformer(l0_points, 7, 128)

        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


    
# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()
    
#     def forward(self, pred, target, trans_feat=None, weight=None):
#         if weight is None:
#             class_counts = torch.bincount(target.flatten(), minlength=pred.size(1))
#             weight = 1.0 / (class_counts.float() + 1e-6)
#             weight = weight / weight.sum()
#             weight = weight.to(pred.device)
#         total_loss = F.nll_loss(pred, target, weight=weight)
#         return total_loss
    
class get_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, trans_feat=None, weight=None):
        return F.nll_loss(pred, target, weight=weight)

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)
        )
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, x, dim):
        assert x.size(2) == dim, f"Expected dim={dim}, but got {x.size(2)}"
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x