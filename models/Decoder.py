import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attention import Self_MultiAttention,FFN,Cross_MultiAttention


'''
    Input:
        __init__:
            dim_q: int
            dim_kv: int
            num_heads: int
            use_self_attn: bool
        forward:
            text_feature: [Batch_size, 1, dim_text]
            gs_feature: [Batch_size, N, dim_gs]
            mask: [Batch_size, N, 1]
    Output:
        forward:
            ffn_feature: [Batch_size, 1, dim_text]
'''
class Transformer_Decoder_Block(nn.Module):
    def __init__(self,
                 dim_q,
                 dim_kv,
                 num_heads,
                 use_self_attn=False,
                 att_drop=0.,
                 lin_drop=0.):
        super(Transformer_Decoder_Block, self).__init__()
        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.sa=Self_MultiAttention(
                dim_q,
                dim_q,
                num_heads,
                lin_before_q=False,
                lin_before_k=False,
                lin_before_v=False,
                lin_after_att=False,
                att_drop=att_drop,
                lin_drop=lin_drop
            )
        self.ca=Cross_MultiAttention(
            dim_q,
            dim_kv,
            num_heads,
            att_drop=att_drop,
            lin_drop=lin_drop,
            lin_before_qkv=True,
            lin_after_att=False
        )
        self.ffn=FFN(dim_q,[256,dim_q])
        self.sa_dropout = nn.Dropout(att_drop)
        self.ca_dropout = nn.Dropout(att_drop)
        self.ffn_dropout= nn.Dropout(lin_drop)

        self.sa_layernorm = nn.LayerNorm(dim_q)
        self.ca_layernorm = nn.LayerNorm(dim_q)
        self.ffn_layernorm = nn.LayerNorm(dim_q)


    def forward(self, text_feature,gs_feature,mask=None):
        if self.use_self_attn:
            sa_feature,_ = self.sa(text_feature,text_feature,text_feature) # [Batch_size, 1, dim_text]
            sa_feature = self.sa_layernorm(text_feature + self.sa_dropout(sa_feature)) # [Batch_size, 1, dim_text]
        else:
            sa_feature = text_feature

        ca_feature,_ = self.ca(sa_feature,gs_feature,gs_feature,mask) # [Batch_size, 1, dim_text]
        ca_feature = self.ca_layernorm(sa_feature + self.sa_dropout(ca_feature)) # [Batch_size, 1, dim_text]

        ffn_feature = self.ffn(ca_feature,mask) # [Batch_size, 1, dim_text]
        ffn_feature = self.ffn_layernorm(ca_feature+self.ffn_dropout(ffn_feature)) # [Batch_size, 1, dim_text]

        return ffn_feature


'''
    Input:
        __init__:
            dim_q: int
            dim_kv: int
            num_heads: int
            decoder_block_num: int
            lin_after_decode: bool
        forward:
            text_feature: [Batch_size, 1, dim_text]
            gs_feature: [Batch_size, N, dim_gs]
            mask: [Batch_size, N]
    Output:
        forward:
            output_features: [Batch_size, 1, dim_text] or [Batch_size, 1, dim_gs]
'''
class Decoder(nn.Module):
    def __init__(self,dim_q,dim_kv,num_heads,decoder_block_num,lin_after_decode=False):
        super(Decoder,self).__init__()
        self.decoder_block_num=decoder_block_num
        self.lin_after_decode=lin_after_decode
        self.decoder_blocks=nn.ModuleList()
        for i in range(decoder_block_num):
            self.decoder_blocks.append(Transformer_Decoder_Block(dim_q,dim_kv,num_heads))

        if lin_after_decode:
            self.last_linear= nn.Linear(dim_q,dim_kv)

    def forward(self, text_features,gs_features,mask=None):
        if mask is not None:
            mask = mask.unsqueeze(-1) # [Batch_size, N, 1]
        output_features=text_features
        for i in range(self.decoder_block_num):
            output_features = self.decoder_blocks[i](output_features,gs_features,mask)

        if self.lin_after_decode:
            output_features = self.last_linear(output_features)
        return output_features