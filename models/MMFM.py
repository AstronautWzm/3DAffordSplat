import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attention import *


# MLP
'''
    Input:
        __init__:
             in_channel: int
             mlp_list: [out_channels]
             active_func: 'relu' or 'sigmoid' or 'tanh'
        forward:
            x: [Batch_size, N, C]
    Output:
        forward:
            x: [Batch_size, N, last(out_channels)]
'''
class MLP(nn.Module):
    def __init__(self,in_channel,mlp_list,active_func='relu'):
        super(MLP, self).__init__()
        self.mlp = nn.ModuleList()
        self.bns = nn.ModuleList()

        if active_func == 'relu':
            self.active_func = nn.ReLU()
        elif active_func == 'sigmoid':
            self.active_func = nn.Sigmoid()
        elif active_func == 'tanh':
            self.active_func = nn.Tanh()
        else:
            raise ValueError('Invalid activation function')

        last_channel = in_channel
        for out_channel in mlp_list:
            self.mlp.append(nn.Linear(last_channel, out_channel))
            self.bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel


    def forward(self, x, mask=None):
        for i in range(len(self.mlp)):
            x = self.mlp[i](x)

            if mask is not None:
                x = x * mask

            x = x.transpose(1, 2)  # [Batch_size, N, C] -> [Batch_size, C, N]
            x = self.bns[i](x)
            x = x.transpose(1, 2)  # [Batch_size, C, N] -> [Batch_size, N, C]
            x = self.active_func(x)

        return x



'''
    Input:
        __init__:
             mlp_list: [out_channels]
             dim_q: int
             dim_kv: int
             num_heads: int
             dim_out: int   (dim_q = dim_kv = mlp_list[-1] = dim_out)
             maxpoints: list
             att_drop: float
             lin_drop: float
             lin_after_att: bool
             layernorm_after_att: bool
        forward:
            Q: [Batch_size, seq_len_q = 1, dim_q]
            K: [Batch_size, seq_len_kv = N, dim_kv]
            V: [Batch_size, seq_len_kv = N, dim_kv]
    Output:
        forward:
            output_feature : [Batch_size, N, dim_out]
'''
class MMFM(nn.Module):
    def __init__(self,
                 mlp_list,
                 dim_q,
                 dim_kv,
                 num_heads,
                 dim_out,
                 maxpoints,
                 att_drop=0.,
                 lin_drop=0.,
                 lin_after_att=False,
                 layernorm_after_att=True,
                 ):
        super(MMFM, self).__init__()
        assert mlp_list[-1] == dim_q == dim_kv, "The setting must be: mlp_list[-1] == dim_q == dim_kv == dim_out"
        self.dim_out = dim_out
        self.dim_kv = dim_kv

        self.cross_att1=Cross_MultiAttention(dim_q,dim_kv,num_heads,dim_q,att_drop,lin_drop,lin_after_att)
        self.mlp=MLP(dim_q,mlp_list)
        if layernorm_after_att:
            self.layernorm=nn.LayerNorm(dim_q)
        else:
            self.layernorm=None


        self.pos_embeds = nn.ModuleDict({
            f"pos_embed_{maxpoint}": nn.Embedding(maxpoint, dim_kv)
            for maxpoint in maxpoints
        })
        self.channel_att1 = Channel_Attention(2 * dim_kv,dim_out)

        if self.dim_kv != self.dim_out:
            self.rest_layer = nn.Linear(dim_kv, dim_out)

    def forward(self, Q, K, V):
        B, N, C = K.shape


        att_output1, att_weights1 = self.cross_att1(Q, K, V) # att_output1: [B, 1, dim_q]


        residual = att_output1 + Q # [B, 1, dim_q]
        if self.layernorm:
            residual = self.layernorm(residual)
        mlp_output1 = self.mlp(residual) # [B, 1, mlp_list[-1]=C]


        self.pos_embed = self.pos_embeds[f"pos_embed_{N}"]
        pos_embedding = self.pos_embed(torch.arange(N).to(K.device))  # [N, C]
        global_feature = mlp_output1.expand(-1, N, -1) + pos_embedding.unsqueeze(0) # [B, N, C]


        fused_feature = torch.cat([global_feature, K], dim=-1)  # [B, N, 2C]


        output_feature = self.channel_att1(fused_feature) # [B, N, dim_out]

        if self.dim_kv != self.dim_out:
            rest_K = self.rest_layer(K) 
        else:
            rest_K = K

        output_feature = output_feature + rest_K # [B, N, dim_out]

        return output_feature


