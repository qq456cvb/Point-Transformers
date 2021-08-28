import torch
import torch.nn as nn
from pointnet_util import PointNetSetAbstractionMsg
from .transformer import MultiHeadAttention


class SortNet(nn.Module):
    def __init__(self, d_model, d_points=6, k=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.sa = PointNetSetAbstractionMsg(k, [0.1, 0.2, 0.4], [16, 32, 128], d_model, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.fc_agg = nn.Sequential(
            nn.Linear(64 + 128 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, d_model - 1 - d_points),
        )
        self.k = k
        self.d_points = d_points
        
    def forward(self, points, features):
        score = self.fc(features)
        topk_idx = torch.topk(score[..., 0], self.k, 1)[1]
        features_abs = self.sa(points[..., :3], features, topk_idx)[1]
        res = torch.cat((self.fc_agg(features_abs),
                         torch.gather(score, 1, topk_idx[..., None].expand(-1, -1, score.size(-1))),
                         torch.gather(points, 1, topk_idx[..., None].expand(-1, -1, points.size(-1)))), -1)
        return res
    
    
class LocalFeatureGeneration(nn.Module):
    def __init__(self, d_model, m, k, d_points=6, n_head=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_points, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        ) 
        self.sortnets = nn.ModuleList([SortNet(d_model, k=k) for _ in range(m)])
        self.att = MultiHeadAttention(n_head, d_model, d_model, d_model // n_head, d_model // n_head)
        
    def forward(self, points):
        x = self.fc(points)
        x, _ = self.att(x, x, x)
        out = torch.cat([sortnet(points, x) for sortnet in self.sortnets], 1)
        return out, x
    

class GlobalFeatureGeneration(nn.Module):
    def __init__(self, d_model, k, d_points=6, n_head=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_points, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        ) 
        self.sa = PointNetSetAbstractionMsg(k, [0.1, 0.2, 0.4], [16, 32, 128], d_model, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.att = MultiHeadAttention(n_head, d_model, d_model, d_model // n_head, d_model // n_head)
        self.fc_agg = nn.Sequential(
            nn.Linear(64 + 128 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, d_model),
        )
        
    def forward(self, points):
        x = self.fc(points)
        x, _ = self.att(x, x, x)
        out = self.fc_agg(self.sa(points[..., :3], x)[1])
        return out, x
    
    
class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model_l, d_model_g, d_reduce, m, k, n_c, d_points, n_head \
            = cfg.model.global_dim, cfg.model.local_dim, cfg.model.reduce_dim, cfg.model.m, cfg.model.k, cfg.num_class, cfg.input_dim, cfg.model.n_head
        self.lfg = LocalFeatureGeneration(d_model=d_model_l, m=m, k=k, d_points=d_points)
        self.gfg = GlobalFeatureGeneration(d_model=d_model_g, k=cfg.model.global_k, d_points=d_points)
        self.lg_att = MultiHeadAttention(n_head, d_model_l, d_model_g, d_model_l // n_head, d_model_l // n_head)
        self.fc = nn.Sequential(
            nn.Linear(d_model_l, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, d_reduce),
        )
        self.fc_cls = nn.Sequential(
            nn.Linear(k * m * d_reduce, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        
    def forward(self, points):
        local_features = self.lfg(points)[0]
        global_features = self.gfg(points)[0]
        lg_features = self.lg_att(local_features, global_features, global_features)[0]
        x = self.fc(lg_features).reshape(points.size(0), -1)
        out = self.fc_cls(x)
        return out
        
