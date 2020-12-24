import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# reference https://github.com/jadore801120/attention-is-all-you-need-pytorch

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, q, k, v):
        attn = q @ k.transpose(-1, -2)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-1)
        output = attn @ v

        return output, attn
        
        
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model_q, d_model_kv, d_k, d_v):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model_q, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model_kv, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model_kv, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model_q, bias=False)

        self.attention = Attention()

        self.layer_norm1 = nn.LayerNorm(n_head * d_v, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model_q, eps=1e-6)


    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        b_size, n_q, n_k = q.size(0), q.size(1), k.size(1)

        residual = q

        # Pass through the pre-attention projection: b x k x (n*dv)
        # Separate different heads: b x k x n x dv
        q = self.w_qs(q).view(-1, n_q, n_head, d_k)
        k = self.w_ks(k).view(-1, n_k, n_head, d_k)
        v = self.w_vs(v).view(-1, n_k, n_head, d_v)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # get b x n x k x dv
        q, attn = self.attention(q, k, v)
        
        # b x k x ndv
        q = q.transpose(1, 2).contiguous().view(b_size, n_q, -1)
        s = self.layer_norm1(residual + q)
        res = self.layer_norm2(s + self.fc(s))

        return res, attn