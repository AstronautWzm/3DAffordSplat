import torch
import torch.nn as nn
import torch.nn.functional as F

# 前馈神经网络
'''
    Input:
        __init__:
            in_channel: int
            out_channel_list: list, [out_channels]
            dropout: float
            active_func: 'relu' or 'sigmoid' or 'tanh'
        forward:
            x: [Batch_size, N, in_channel]
    Output:
        forward:
            x: [Batch_size, N, in_channel]
            mask: None or [Batch_size, N, 1]
'''
class FFN(nn.Module):
    def __init__(self,in_channel,out_channel_list,dropout=0.,active_func='relu'):
        super(FFN,self).__init__()
        self.ffn=nn.ModuleList()
        last_channel=in_channel

        if active_func == 'relu':
            self.active_func = nn.ReLU()
        elif active_func == 'sigmoid':
            self.active_func = nn.Sigmoid()
        elif active_func == 'tanh':
            self.active_func = nn.Tanh()
        else:
            raise ValueError('Invalid activation function')

        if out_channel_list[-1] != in_channel:
            raise ValueError('out_channel_list[-1] != in_channel')

        for out_channel in out_channel_list:
            self.ffn.append(nn.Linear(last_channel,out_channel))
            if out_channel != out_channel_list[-1]:
                self.ffn.append(self.active_func)
                self.ffn.append(nn.Dropout(dropout))
            last_channel=out_channel

        self.ffn=nn.Sequential(*self.ffn)

    def forward(self,x,mask=None):
        if mask is None:
            x=self.ffn(x)
        else:
            x = x * mask
            x = self.ffn(x)
            x = x * mask
        return x


# 自多头注意力
'''
    Input:
        __init__:
            dim_q: int
            dim_kv: int
            num_heads: int
            dim_out: int
            lin_before_q: bool
            lin_before_k: bool
            lin_before_v: bool
            lin_after_att: bool
            att_drop: float
            lin_drop: float
        forward:
            Q: [Batch_size, seq_len_q, dim_q]
            K: [Batch_size, seq_len_kv, dim_kv]
            V: [Batch_size, seq_len_kv, dim_kv]
            att_mask: [Batch_size, N, 1]
    Output:
        forward:
            att_output: [Batch_size, seq_len_q, dim_q] or [Batch_size, seq_len_q, dim_out]
            att_weights: [Batch_size, num_heads, seq_len_q, seq_len_kv]
'''
class Self_MultiAttention(nn.Module):
    def __init__(self,
                 dim_q,
                 dim_kv,
                 num_heads,
                 dim_out=256,
                 lin_before_q=True,
                 lin_before_k=True,
                 lin_before_v=True,
                 lin_after_att=True,
                 att_drop=0.,
                 lin_drop=0.):
        super(Self_MultiAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.att_drop = nn.Dropout(att_drop)
        self.lin_drop = nn.Dropout(lin_drop)
        assert dim_q % num_heads == 0, "The Query dimension must be divisible by the Number of Heads"
        self.dim_head = dim_q // num_heads

        self.W_q = nn.Linear(dim_q, dim_q)
        if lin_before_q:
            self.W_q = nn.Linear(dim_q, dim_q)
        else:
            self.W_q = None
        if lin_before_k:
            self.W_k = nn.Linear(dim_kv, dim_q)
        else:
            self.W_k = None
        if lin_before_v:
            self.W_v = nn.Linear(dim_kv, dim_q)
        else:
            self.W_v = None
        if lin_after_att:
            self.W_o = nn.Sequential(nn.Linear(dim_q, dim_out), self.lin_drop)
        else:
            self.W_o = None

    def scaled_dot_product_attention(self,Q, K, V, att_mask):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim_head, dtype=Q.dtype))

        # 应用注意力掩码，忽略填充部分
        if att_mask is not None:
            att_mask = att_mask.bool()
            scores = scores.masked_fill(att_mask, -1e9)

        att_weights = F.softmax(scores, dim=-1) # (batch_size, num_heads, seq_len_q, seq_len_kv)
        att_weights = self.att_drop(att_weights)
        att_output = torch.matmul(att_weights, V) # (batch_size, num_heads, seq_len_q, dim_head)
        return att_output, att_weights

    def forward(self, Q, K, V, att_mask=None):
        if att_mask is not None:
            att_mask = att_mask.permute(0, 2, 1) # (batch_size, 1, N)
            att_mask = att_mask[:, None,:] # # (batch_size, 1, 1, N)
        batch_size, seq_len, dim_q = Q.size()
        assert dim_q == self.dim_q, "The input Query dimension must be the same as dim_q in function __init__()"

        if self.W_q:
            Q = self.W_q(Q) # (batch_size, seq_len_kv, dim_kv) -> (batch_size, seq_len_kv, dim_q)

        if self.W_k:
            K = self.W_k(K) # (batch_size, seq_len_kv, dim_kv) -> (batch_size, seq_len_kv, dim_q)
        else:
            assert K.size(-1) == dim_q, "The input Key dimension must be the same as dim_q in function __init__()"

        if self.W_v:
            V = self.W_v(V) # (batch_size, seq_len_kv, dim_kv) -> (batch_size, seq_len_kv, dim_q)
        else:
            assert V.size(-1) == dim_q, "The input Value dimension must be the same as dim_q in function __init__()"

        Q = Q.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2) # (batch_size, seq_len_q, dim_q) -> (batch_size, num_heads, seq_len_q, dim_head)
        K = K.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2) # (batch_size, seq_len_kv, dim_q) -> (batch_size, num_heads, seq_len_kv, dim_head)
        V = V.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2) # (batch_size, seq_len_kv, dim_q) -> (batch_size, num_heads, seq_len_kv, dim_head)

        att_output, att_weights = self.scaled_dot_product_attention(Q, K, V, att_mask)
        att_output = att_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_head) # (batch_size, num_heads, seq_len_q, dim_head) -> (batch_size, seq_len_q, dim_q)

        if self.W_o:
            att_output = self.W_o(att_output)

        return att_output, att_weights


# 多头交叉注意力
'''
    Input:
        __init__:
             dim_q: int
             dim_kv: int
             num_heads: int
             dim_out: int
             att_drop: float
             lin_drop: float
             lin_before_qkv: bool
             lin_after_att: bool
        forward:
            Q: [Batch_size, seq_len_q, dim_q]
            K: [Batch_size, seq_len_kv, dim_kv]
            V: [Batch_size, seq_len_kv, dim_kv]
            att_mask: [Batch_size, N, 1]
    Output:
        forward:
            att_output: [Batch_size, seq_len_q, dim_q] or [Batch_size, seq_len_q, dim_out]
            att_weights: [Batch_size, num_heads, seq_len_q, seq_len_kv]
'''
class Cross_MultiAttention(nn.Module):
    def __init__(self,
                 dim_q,
                 dim_kv,
                 num_heads,
                 dim_out=256,
                 att_drop=0.,
                 lin_drop=0.,
                 lin_before_qkv=True,
                 lin_after_att=True):
        super(Cross_MultiAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.att_drop = nn.Dropout(att_drop)
        self.lin_drop = nn.Dropout(lin_drop)
        assert dim_q % num_heads == 0, "The Query dimension must be divisible by the Number of Heads"
        self.dim_head = dim_q // num_heads
        self.lin_after_att = lin_after_att
        self.lin_before_qkv = lin_before_qkv

        if self.lin_before_qkv:
            self.W_q = nn.Linear(dim_q, dim_q)
            self.W_k = nn.Linear(dim_kv, dim_q)
            self.W_v = nn.Linear(dim_kv, dim_q)
        self.W_o = nn.Sequential(nn.Linear(dim_q, dim_out), self.lin_drop)


    def scaled_dot_product_attention(self,Q, K, V, att_mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim_head, dtype=Q.dtype)) # (batch_size, num_heads, seq_len_q, seq_len_kv)

        # 应用注意力掩码，忽略填充部分
        if att_mask is not None:
            att_mask = att_mask.bool()
            scores = scores.masked_fill(att_mask, -1e9)

        att_weights = F.softmax(scores, dim=-1) # (batch_size, num_heads, seq_len_q, seq_len_kv)

        att_weights = self.att_drop(att_weights)
        att_output = torch.matmul(att_weights, V) # (batch_size, num_heads, seq_len_q, dim_head)
        return att_output, att_weights


    def forward(self, Q, K, V, att_mask=None):
        if att_mask is not None:
            att_mask = att_mask.permute(0, 2, 1) # (batch_size, 1, N)
            att_mask = att_mask[:, None, : ,:] # (batch_size, 1, 1, N)
        batch_size, seq_len, dim_q = Q.size()
        assert dim_q == self.dim_q, "The input Query dimension must be the same as dim_q in function __init__()"

        if self.lin_before_qkv:
            Q = self.W_q(Q) # (batch_size, seq_len_q, dim_q) -> (batch_size, seq_len_q, dim_q)
            K = self.W_k(K) # (batch_size, seq_len_kv, dim_kv) -> (batch_size, seq_len_kv, dim_q)
            V = self.W_v(V) # (batch_size, seq_len_kv, dim_kv) -> (batch_size, seq_len_kv, dim_q)

        Q = Q.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2) # (batch_size, seq_len_q, dim_q) -> (batch_size, num_heads, seq_len_q, dim_head)
        K = K.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2) # (batch_size, seq_len_kv, dim_q) -> (batch_size, num_heads, seq_len_kv, dim_head)
        V = V.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2) # (batch_size, seq_len_kv, dim_q) -> (batch_size, num_heads, seq_len_kv, dim_head)

        att_output, att_weights = self.scaled_dot_product_attention(Q, K, V, att_mask)
        att_output = att_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_head) # (batch_size, num_heads, seq_len_q, dim_head) -> (batch_size, seq_len_q, dim_q)

        if self.lin_after_att:
            att_output = self.W_o(att_output)

        return att_output, att_weights


# 通道注意力
'''
    Input:
        __init__:
             in_channels: int
             out_channels: int
             reduction_ratio: int
        forward:
            x: [Batch_size, N, in_channels]
    Output:
        forward:
            att_output: [Batch_size, N, out_channels]
'''
class Channel_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(in_channels, out_channels)  # 维度投影

    def forward(self, x):
        Batch_size, N, in_channels = x.shape
        # 通道注意力权重
        channel_weights = self.avg_pool(x.transpose(1, 2))  # [Batch_size, in_channels, 1]
        channel_weights = self.fc(channel_weights.squeeze(-1))  # [Batch_size, in_channels]
        channel_weights = channel_weights.unsqueeze(1)  # [Batch_size, 1, in_channels]
        # 加权并投影
        weighted_x = x * channel_weights # [Batch_size, N, in_channels]
        return self.proj(weighted_x)  # [B, N, out_channels]