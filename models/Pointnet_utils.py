import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.xpu import device

from .Attention import Self_MultiAttention,FFN


"""
    Input:
        points: [N, C]
        nsample: int
    Output:
        centroids: [nsample,C]
"""
def fps_one(points,nsample):
    N, C = points.shape
    centroids = np.zeros((nsample, C))
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(nsample):
        centroids[i] = points[farthest]
        dist = np.sum((points[:,:3] - points[farthest,:3]) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)

    return centroids



"""
    Input:
        points: [Batch_size, N, C]
        nsample: int
        device："cuda" or "cpu"
    Output:
        centroids: [Batch_size, nsample, C]
"""
def fps_batch(points, nsample, device):
    B, N, C = points.shape
    centroids = torch.zeros(B, nsample, C, device=device, dtype=points.dtype)
    distance = torch.ones(B, N, device=device, dtype=points.dtype) * 1e10
    farthest = torch.randint(0, N, (B,), device=device)
    batch_indices = torch.arange(B, device=device)

    for i in range(nsample):
        centroids[:, i] = points[batch_indices, farthest]
        dist = torch.sum((points[:, :, :3] - points[batch_indices, farthest].unsqueeze(1)[:, :, :3]) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, dim=1)

    return centroids


'''
    Input:
        points: [Batch_size, N, C]
        query_points：[Batch_size, nsample, C]
        k：int
    Output:
        grouped_points: [Batch_size, nsample, k, C]
'''
def knn_groups(points, query_points, k):
    batch_size, npoints, C = points.shape
    _, nsample, _ = query_points.shape
    dist = torch.cdist(query_points[:, :, :3], points[:, :, :3], p=2) # [Batch_size, nsample, npoints]
    min_k = min(k, npoints)
    _, indices_min = torch.topk(dist, min_k, dim=-1, largest=False) # [Batch_size, nsample, min_k]

    if min_k < k:

        times = (k + min_k - 1) // min_k
        indices_expanded = indices_min.repeat_interleave(times, dim=2)[:, :, :k] # [Batch_size, nsample, k]
    else:
        indices_expanded = indices_min

    batch_size, nsample, k_final = indices_expanded.shape
    points_expanded = points.unsqueeze(1).expand(batch_size, nsample, npoints, C)
    indices_expanded_unsqueezed = indices_expanded.unsqueeze(-1).expand(batch_size, nsample, k_final, C)

    grouped_points = torch.gather(points_expanded, 2, indices_expanded_unsqueezed) # [Batch_size, nsample, k, C]
    return grouped_points


# After upsampling the AffordSplat (fps), it is grouped (knn) and then subjected to feature extraction (conv);
# Using MSG strategy to handle uneven sampling
'''
    Input:
        __init__:
            nsample: int
            knn_points: [ngroups]
            in_channel：int
            mlp_list：[nlayers, out_channels]
        forward:
            points：[Batch_size, N, C]
    Output:
        forward:
            features:[Batch_size, nsample, sum(out_channels)]
'''
class Pointnet_SetAbstraction_msg(nn.Module):
    def __init__(self,nsample,knn_points,in_channel,mlp_list):
        super(Pointnet_SetAbstraction_msg,self).__init__()
        self.nsample = nsample
        self.knn_points = knn_points
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for block in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel=in_channel
            for out_channel in mlp_list[block]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel=out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, points):
        device=points.device
        features_list=[]
        for i,knn_point in enumerate(self.knn_points):
            centroids=fps_batch(points,self.nsample,device)
            grouped_points=knn_groups(points,centroids,knn_point)
            grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous() # [Batch_size, nsample, k, C] -> [Batch_size, C, nsample, k]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points))) # [Batch_size, C', nsample, k]
            pooled_features = torch.max(grouped_points, dim=-1)[0] # [Batch_size, C', nsample]
            features_list.append(pooled_features)
        features = torch.cat(features_list, dim=1) # [Batch_size, sum(C'), nsample]
        features = features.permute(0, 2, 1).contiguous()  # [Batch_size, nsample, sum(C')]
        return features


# Transformer_Encoder_block
'''
    Input:
        __init__:
            ffn_list: list, [out_channels]
            dim_q: int
            dim_kv: int
            num_heads: int
            att_drop: float
            lin_drop: float
        forward:
            x: [Batch_size, N, in_channel]
    Output:
        forward:
            x: [Batch_size, N, in_channel]
'''
class Transformer_Encoder_Block(nn.Module):
    def __init__(self,
                 ffn_list,
                 dim_q,
                 dim_kv,
                 num_heads,
                 att_drop=0.,
                 lin_drop=0.,
                 ):
        super(Transformer_Encoder_Block, self).__init__()
        self.att = Self_MultiAttention(dim_q,dim_kv,num_heads,att_drop=att_drop,lin_drop=lin_drop,lin_before_v=False,lin_after_att=False)
        self.ffn = FFN(dim_q,ffn_list)
        self.layer_norm1 = nn.LayerNorm(dim_q)
        self.layer_norm2 = nn.LayerNorm(dim_q)

    def forward(self, x):
        att_output, att_weights = self.att(x, x, x)
        middle = att_output + x
        middle = self.layer_norm1(middle)
        output = self.ffn(middle)
        output = output + middle
        output = self.layer_norm2(output)
        return output


# Pointnet_SetAbstraction_msg with transformer encoder
'''
    Input:
        __init__:
            nsample: int
            knn_points: [ngroups]
            in_channel: int
            mlp_list: [nlayers, out_channels]
            ffn_list: list, [sum(out_channels)]
            num_heads: int
            att_drop: float
            lin_drop: float
            transformer_encoder_num: int
        forward:
            points：[Batch_size, N, C]
    Output:
        forward:
            features: [Batch_size, nsample, sum(out_channels)+3]
            centroids_xyz:[Batch_size, nsample, 3]
'''
class Pointnet_with_transformer(nn.Module):
    def __init__(self,nsample,knn_points,in_channel,mlp_list,ffn_list,num_heads,att_drop=0.,lin_drop=0.,transformer_encoder_num=1):
        super(Pointnet_with_transformer, self).__init__()
        self.nsample = nsample
        self.knn_points = knn_points
        self.transformer_encoder_num = transformer_encoder_num
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.transformer_encoder = nn.ModuleList()

        for block in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp_list[block]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel=out_channel+3
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

        self.dim_in=sum(block[-1] for block in mlp_list)+3
        for i in range(transformer_encoder_num):
            self.encoder_block = Transformer_Encoder_Block(ffn_list, self.dim_in, self.dim_in, num_heads, att_drop,lin_drop)
            self.transformer_encoder.append(self.encoder_block)


    def forward(self, points):
        device=points.device
        features_list=[]
        centroids = fps_batch(points, self.nsample, device)
        centroids_xyz = centroids[:, :, :3] # [Batch_size, nsample, 3]

        for i,knn_point in enumerate(self.knn_points):
            grouped_points=knn_groups(points,centroids,knn_point)
            grouped_points_xyz = grouped_points[:, :, :, :3] # [Batch_size, nsample, k, 3]
            grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous() # [Batch_size, nsample, k, C] -> [Batch_size, C, nsample, k]

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points))) # [Batch_size, C', nsample, k]
                
                if j != len(self.conv_blocks[i])-1:
                    grouped_points = grouped_points.permute(0, 2, 3, 1).contiguous() # [Batch_size, nsample, k, C']
                    grouped_points = torch.cat((grouped_points,grouped_points_xyz),dim=-1) # [Batch_size, nsample, k, C'+3]
                    grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous() # [Batch_size, C'+3, nsample, k]
            pooled_features = torch.max(grouped_points, dim=-1)[0] # [Batch_size, C', nsample]
            features_list.append(pooled_features)

        features = torch.cat(features_list, dim=1) # [Batch_size, sum(C'), nsample]
        features = features.permute(0, 2, 1).contiguous()  # [Batch_size, nsample, sum(C')]
        features = torch.cat((features,centroids_xyz),dim=-1)
        for i in range(self.transformer_encoder_num):
            encoder=self.transformer_encoder[i]
            features=encoder(features)

        return features,centroids_xyz


