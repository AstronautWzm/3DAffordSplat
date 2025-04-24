import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Attention import *
from models.MMFM import MLP
from models.Pointnet_utils import *


# 二值化掩码
# 使用 Straight-Through Estimator (STE) 保留梯度
'''
    Input:
        aff_map: [Batch_size, N, 1]
        threshold: float
    Output:
        Binary_mask: [Batch_size, N, 1]
'''
def Binary_mask(aff_map, threshold=0.5, max_pc_ratio=None,min_coverage=0.1):
    B, N, _ = aff_map.shape  # Batch_size, Num_points, 1

    base_mask = (aff_map > threshold).float()
    valid_batch = (base_mask.sum(dim=1, keepdim=True) > 0).float()  # [Batch_size, 1, 1]
    adjust_mask = (1 - valid_batch).bool()  # 需要调整阈值的样本标记

    # 动态阈值计算（保证至少前min_coverage比例的点）
    sorted_values = torch.sort(aff_map, dim=1, descending=True).values  # [B, N, 1]
    k_min  = max(1, int(min_coverage * N))
    dynamic_threshold_min = sorted_values[:, k_min  - 1:k_min , :]  # 取第k大的值作为动态阙值 # [B, 1, 1]

    dynamic_threshold_max = None
    adjust_mask_max = torch.zeros_like(valid_batch, dtype=torch.bool)

    if max_pc_ratio is not None:
        # 完整模型点数
        valid_points = (aff_map > 0).sum(dim=1, keepdim=True).float()  # [B,1,1]
        # 允许的最大点数（向上取整，至少1，且不超过总点数）
        allowed_high = (valid_points * max_pc_ratio).ceil().clamp(min=1, max=N) # [B,1,1]
        allowed_high = allowed_high.to(torch.int64)  # 确保是int64类型
        # 获取动态阈值（第allowed_high大的值）
        indices = (allowed_high - 1).clamp(min=0)  # 索引从0开始
        dynamic_threshold_max = torch.gather(sorted_values, 1, indices)  # [B,1,1]
        # 检查当前是否超过max_pc_ratio限制
        current_high = base_mask.sum(dim=1, keepdim=True)  # 当前满足阈值的点数
        exceeds_max_ratio = (current_high > allowed_high.float()).bool()  # [B,1,1]
        adjust_mask_max = exceeds_max_ratio # [B, 1, 1]

    # 混合阈值
    final_threshold = torch.where(adjust_mask,
                                  dynamic_threshold_min,
                                  torch.full_like(dynamic_threshold_min, threshold)) # [Batch_size, 1, 1]
    if max_pc_ratio is not None:
        final_threshold = torch.where(adjust_mask_max, dynamic_threshold_max, final_threshold)

    final_threshold = final_threshold.expand(-1, N, -1)  # [Batch_size, N, 1]

    # 生成最终掩码
    hard_mask = (aff_map >= final_threshold).float()

    # STE梯度保留
    adjust_total = adjust_mask #| adjust_mask_max # [B, 1, 1]
    soft_mask = aff_map * (1 + 1 * adjust_total.float())  # 增强梯度 # [Batch_size, N, 1]
    binary_mask = hard_mask.detach() + (soft_mask - soft_mask.detach()) # [Batch_size, N, 1]

    return binary_mask


# 考虑掩码的批归一化
# 输入数据包含大量非标注点（掩码值为0的区域）。传统的批归一化会统计所有点的均值和方差，导致非标注点的零值干扰统计量计算。
'''
    Input:
        __init__:
             in_channel: int
        forward:
             x: [Batch_size, C, N, k]
             aff_map: [Batch_size, N, 1]
    Output:
        forward:
             output: [Batch_size, C, N, k]
'''
class MaskedBatchNorm2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        # 禁用running统计量，关闭仿射变换（后续手动实现）
        self.bn = nn.BatchNorm2d(
            in_channel,
            affine=False,
            track_running_stats=False
        )
        # 自定义可学习的仿射参数
        self.weight = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.momentum = 0.1

    def forward(self, x, aff_map=None):
        if aff_map is None:
            return self.bn(x)

        B, C, N, k = x.shape
        aff_map = aff_map.view(B, 1, N, 1)  # [B,1,N,1]

        # 有效元素统计（避免零掩码干扰）
        valid_count = aff_map.sum(dim=[0, 2, 3], keepdim=True) + 1e-5  # [1,C,1,1]

        # 动态计算当前batch统计量
        masked_x = x * aff_map
        sum_x = masked_x.sum(dim=[0, 2, 3], keepdim=True)
        mean = sum_x / valid_count

        # 方差计算（带掩码广播）
        var = ((x - mean) ** 2 * aff_map).sum(dim=[0, 2, 3], keepdim=True) / valid_count

        # 归一化并应用仿射变换
        x_normalized = (x - mean) / torch.sqrt(var + 1e-5)
        return x_normalized * self.weight + self.bias


# 考虑掩码的卷积
'''
    Input:
        __init__:
             in_channel: int
             out_channel: int
        forward:
            x: [Batch_size, C, N, k]
            aff_map: [Batch_size, N, 1]
    Output:
        forward:
            output: [Batch_size, C, N, k]
'''
class MaskedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel,kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size)

    def forward(self, x, aff_map=None):
        if aff_map is not None:# 动态抑制非标注点
            x = x * aff_map.unsqueeze(1)  # [Batch_size, C, N, k]
        return self.conv(x)


# 同时对高斯模型和高斯掩码进行fps采样
"""
    Input:
        gs: [N, 10]
        aff_mask: [N, 1]
        nsample: int
    Output:
        sampled_points: [N, 10]
        sampled_mask: [N, 1]
"""
def fps_one_both(gs, aff_mask, nsample):
    N, C = gs.shape
    device = gs.device
    sampled_points = torch.zeros((nsample, C), dtype=gs.dtype, device=device) # 存储采样点
    sampled_mask = torch.zeros((nsample, 1), dtype=aff_mask.dtype, device=device) # 存储aff掩码
    distance = torch.full((N,), 1e10, dtype=torch.float32, device=device)
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device).item()

    for i in range(nsample):
        sampled_points[i] = gs[farthest]
        sampled_mask[i] = aff_mask[farthest]

        dist = torch.sum((gs[:, :3] - gs[farthest, :3]) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = torch.argmax(distance).item()

    return sampled_points, sampled_mask


# 考虑聚类邻域内存在填充点或非标注点的knn算法
'''
    Input:
        points: [Batch_size, N, C]
        query_points：[Batch_size, nsample, C]
        mask: [Batch_size, N, 1]
        k：int
    Output:
        grouped_points: [Batch_size, nsample, k, C]
'''
def knn_groups_mask(points, query_points, mask, k):
    batch_size, npoints, C = points.shape
    _, nsample, _ = query_points.shape

    mask_bool = mask.bool()
    far_point = torch.tensor([1e9, 1e9, 1e9], device=points.device).view(1, 1, 3)
    far_point = far_point.expand(batch_size, npoints, 3)  # [Batch_size, N, 3]

    points_coords = torch.where(mask_bool, points[:, :, :3], far_point) # [Batch_size, N, 3]
    dist = torch.cdist(query_points[:, :, :3], points_coords, p=2)
    _, indices = torch.topk(dist, k, dim=-1, largest=False)

    points_expanded = points.unsqueeze(1).expand(batch_size, nsample, npoints, C)
    indices_expanded = indices.unsqueeze(-1).expand(batch_size, nsample, k, C)
    grouped_points = torch.gather(points_expanded, 2, indices_expanded)

    return grouped_points


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
            q_aff_mask: [Batch_size, seq_len_q, 1]（概率掩码）
            q_binary_mask: [Batch_size, seq_len_q, 1]（二值掩码）
            kv_mask: [Batch_size, seq_len_kv, 1]
    Output:
        forward:
            att_output: [Batch_size, seq_len_q, dim_q] or [Batch_size, seq_len_q, dim_out]
            att_weights: [Batch_size, num_heads, seq_len_q, seq_len_kv]
'''
class Cross_MultiAttention_Q_masked(nn.Module):
    def __init__(self,
                 dim_q,
                 dim_kv,
                 num_heads,
                 dim_out=256,
                 att_drop=0.,
                 lin_drop=0.,
                 lin_before_qkv=True,
                 lin_after_att=True):
        super(Cross_MultiAttention_Q_masked, self).__init__()
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

    def scaled_dot_product_attention(self, Q, K, V, q_aff_mask, q_binary_mask, kv_mask):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim_head, dtype=Q.dtype)) # (batch_size, num_heads, seq_len_q, seq_len_kv)

        # 应用注意力掩码，忽略填充部分
        if kv_mask is not None:
            kv_mask = kv_mask.permute(0, 2, 1)  # (batch_size, 1, seq_len_kv)
            kv_mask = kv_mask[:, None, :]  # # (batch_size, 1, 1, seq_len_kv)
            kv_mask = kv_mask.bool()
            scores = scores.masked_fill(kv_mask, -1e9)

        if q_aff_mask is not None: # 将掩码中的概率值结合到注意力权重中
            q_aff_mask = q_aff_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores * q_aff_mask
        if q_binary_mask is not None: # 忽略非标注点部分
            q_binary_mask = q_binary_mask.bool().unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(q_binary_mask, -1e9)

        att_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len_q, seq_len_kv)
        att_weights = self.att_drop(att_weights)
        att_output = torch.matmul(att_weights, V)  # (batch_size, num_heads, seq_len_q, dim_head)
        return att_output, att_weights

    def forward(self, Q, K, V, q_aff_mask=None, q_binary_mask=None,kv_mask=None):
        batch_size, seq_len, dim_q = Q.size()
        assert dim_q == self.dim_q, "The input Query dimension must be the same as dim_q in function __init__()"

        if self.lin_before_qkv:
            Q = self.W_q(Q)  # (batch_size, seq_len_q, dim_q) -> (batch_size, seq_len_q, dim_q)
            K = self.W_k(K)  # (batch_size, seq_len_kv, dim_kv) -> (batch_size, seq_len_kv, dim_q)
            V = self.W_v(V)  # (batch_size, seq_len_kv, dim_kv) -> (batch_size, seq_len_kv, dim_q)

        Q = Q.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)  # (batch_size, seq_len_q, dim_q) -> (batch_size, num_heads, seq_len_q, dim_head)
        K = K.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)  # (batch_size, seq_len_kv, dim_q) -> (batch_size, num_heads, seq_len_kv, dim_head)
        V = V.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)  # (batch_size, seq_len_kv, dim_q) -> (batch_size, num_heads, seq_len_kv, dim_head)

        att_output, att_weights = self.scaled_dot_product_attention(Q, K, V, q_aff_mask, q_binary_mask, kv_mask)
        att_output = att_output.transpose(1, 2).contiguous().view(batch_size, -1,self.num_heads * self.dim_head)  # (batch_size, num_heads, seq_len_q, dim_head) -> (batch_size, seq_len_q, dim_q)

        if self.lin_after_att:
            att_output = self.W_o(att_output)

        return att_output, att_weights


# 对模型进行编码，实现统一维度；单纯使用MLP对每个点的维度进行变化无法学习到任何信息
'''
    Input:
        __init__:
            nsample: int
            knn_points: list
            in_channel: int
            mlp_list: list
            embed_dim: int
        forward:
            points: [Batch_size, max_points_num, 10]
            points_aff_map: [Batch_size, max_points_num, 1]
            mask: [Batch_size, max_points_num]
    Output:
        forward:
            points_features: [Batch_size, nsample, embed_dim]
            aff_features: [Batch_size, nsample, embed_dim]（只对有效部分进行编码）
            sampled_points_aff_map_batch: [Batch_size, nsample, 1]
'''
class gs_encoder(nn.Module):
    def __init__(self, nsample, knn_points, in_channel, mlp_list, embed_dim):
        super(gs_encoder, self).__init__()
        self.nsample = nsample
        self.knn_points = knn_points
        self.C = in_channel
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for block in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp_list[block]:
                convs.append(MaskedConv2d(last_channel, out_channel, 1))
                bns.append(MaskedBatchNorm2d(out_channel))
                last_channel = out_channel + self.C
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

        self.embed_layer = nn.Linear(sum(block[-1] for block in mlp_list), embed_dim)

    def forward(self, points, points_aff_map, mask, threshold=0.5,max_pc_ratio=None):
        device = points.device
        points_features_list = []
        aff_features_list = []

        # 对于高斯特征进行下采样，减少计算,同时得到对应采样点的aff map（不需要掩码，因为采样的点数小于模型的实际点数）
        batch_size = points.size(0)
        # 存储采样结果
        sampled_points_batch = torch.zeros((batch_size, self.nsample, self.C)).to(device)
        sampled_points_aff_map_batch = torch.zeros((batch_size, self.nsample, 1)).to(device)
        for i in range(batch_size):
            actual_points = points[i, mask[i].bool(), :]  # [N,10] # 高斯实际点数（去填充）
            actual_aff_mask = points_aff_map[i, mask[i].bool(), :]  # [N,1] # aff map实际点数（去填充）
            sampled_points, sampled_aff_mask = fps_one_both(actual_points, actual_aff_mask, self.nsample)
            sampled_points_batch[i, :, :] = sampled_points  # [batch_size, nsample, 10] # 采样后的模型
            sampled_points_aff_map_batch[i, :, :] = sampled_aff_mask  # [batch_size, nsample, 1] # 采样点对应的 aff map

        # 采样后的Aff模型
        binary_sampled_points_batch = Binary_mask(sampled_points_aff_map_batch,threshold,max_pc_ratio) # [batch_size, nsample, 1]
        sampled_aff = sampled_points_batch * binary_sampled_points_batch  # [batch_size, nsample, 10]

        # 原Aff模型(包含填充点和非标注点)
        binary_points_aff_map = Binary_mask(points_aff_map,threshold,max_pc_ratio) # [Batch_size, N, 1]
        points_aff = points * binary_points_aff_map  # [Batch_size, max_points_num, 10]

        for i, knn_point in enumerate(self.knn_points):
            grouped_points = knn_groups_mask(points, sampled_points_batch, mask.unsqueeze(-1), knn_point)  # [Batch_size, nsample, k, 10] # 原模型
            grouped_aff = knn_groups_mask(points_aff, sampled_aff, binary_points_aff_map, knn_point) * binary_sampled_points_batch.unsqueeze(-1)  # [Batch_size, nsample, k, 10]
            grouped_points = grouped_points.permute(0, 3, 1,2).contiguous()  # [Batch_size, nsample, k, C] -> [Batch_size, C, nsample, k]
            grouped_aff = grouped_aff.permute(0, 3, 1, 2).contiguous()

            # 对高斯模型进行编码
            grouped_points_features = grouped_points
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points_features = F.relu(bn(conv(grouped_points_features)), inplace=True)  # [Batch_size, C', nsample, k]

                if j != len(self.conv_blocks[i]) - 1:
                    grouped_points_features = torch.cat((grouped_points_features, grouped_points),
                                                        dim=1)  # [Batch_size, C'+C, nsample, k]

            pooled_points_features = torch.max(grouped_points_features, dim=-1)[0]  # [Batch_size, C', nsample]
            points_features_list.append(pooled_points_features)

            # 对高斯Aff模型进行编码
            grouped_aff_features = grouped_aff
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_aff_features = F.relu(bn(conv(grouped_aff_features,binary_sampled_points_batch),binary_sampled_points_batch), inplace=True)  # [Batch_size, C', nsample, k]

                if j != len(self.conv_blocks[i]) - 1:
                    grouped_aff_features = torch.cat((grouped_aff_features, grouped_aff),dim=1)  # [Batch_size, C'+C, nsample, k]

            pooled_aff_features = torch.max(grouped_aff_features, dim=-1)[0]  # [Batch_size, C', nsample]
            aff_features_list.append(pooled_aff_features)

        points_features = torch.cat(points_features_list, dim=1)  # [Batch_size, sum(C'), nsample]
        points_features = points_features.permute(0, 2, 1).contiguous()  # [Batch_size, nsample, sum(C')]
        points_features = self.embed_layer(points_features)  # [Batch_size, nsample, embed_dim]

        aff_features = torch.cat(aff_features_list, dim=1)  # [Batch_size, sum(C'), nsample]
        aff_features = aff_features.permute(0, 2, 1).contiguous()  # [Batch_size, nsample, sum(C')]
        aff_features = self.embed_layer(aff_features)  # [Batch_size, nsample, embed_dim]
        aff_features *= binary_sampled_points_batch # [Batch_size, nsample, embed_dim]

        return points_features, aff_features, sampled_points_aff_map_batch


# 对于点云进行编码（无下采样，需要考虑到pc_aff_map的0，1特性）
'''
    Input:
        __init__:
            knn_points: list
            in_channel: int
            mlp_list: list
            embed_dim: int
        forward:
            points: [Batch_size,pc_num,points_num,3]
            points_aff_map: [Batch_size,pc_num,points_num,1]
    Output:
        forward:
            points_features: [Batch_size,pc_num,points_num,embed_dim]
            aff_features: [Batch_size,pc_num,points_num,embed_dim]
            sampled_points_aff_map_batch: [Batch_size,pc_num,points_num,1]
'''
class pc_encoder(nn.Module):
    def __init__(self, nsample,knn_points, in_channel, mlp_list, embed_dim):
        super(pc_encoder, self).__init__()
        self.nsample = nsample
        self.knn_points = knn_points
        self.C = in_channel
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for block in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp_list[block]:
                convs.append(MaskedConv2d(last_channel, out_channel, 1))
                bns.append(MaskedBatchNorm2d(out_channel))
                last_channel = out_channel + self.C
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

        self.embed_layer = nn.Linear(sum(block[-1] for block in mlp_list), embed_dim)

    def forward(self, points, points_aff_map, threshold=0.5):
        device = points.device
        Batch_size, pc_num, points_num, pc_dim = points.shape

        # 二值化点云掩码
        points_aff_map = points_aff_map.view(Batch_size*pc_num,points_num,-1) # [Batch_size*pc_num, points_num, 1]
        binary_points_aff_map = Binary_mask(points_aff_map, threshold) # [Batch_size*pc_num,points_num,1]

        # 生成 Aff 模型
        points = points.view(Batch_size * pc_num, points_num, pc_dim)  # [Batch_size*pc_num, points_num, 3]
        points_aff = points * binary_points_aff_map # [Batch_size*pc_num, points_num, 3]

        # 对点云进行采样
        if self.nsample > 0:
            points_num = self.nsample
            # 存储采样结果
            sampled_points_batch = torch.zeros((Batch_size * pc_num, self.nsample, self.C)).to(device) # [Batch_size*pc_num, points_num, 3]
            sampled_points_aff_map_batch = torch.zeros((Batch_size * pc_num, self.nsample, 1)).to(device) # [Batch_size*pc_num,points_num,1]
            for i in range(Batch_size * pc_num):
                sampled_points, sampled_aff_mask = fps_one_both(points[i], points_aff_map[i], self.nsample)
                sampled_points_batch[i, :, :] = sampled_points  # [Batch_size*pc_num, nsample, 3] # 采样后的模型
                sampled_points_aff_map_batch[i, :, :] = sampled_aff_mask  # [Batch_size*pc_num,nsample,1] # 采样点对应的 aff map

            # 采样后的Aff模型
            binary_sampled_points_batch = Binary_mask(sampled_points_aff_map_batch,threshold) # [Batch_size*pc_num,nsample,1]
            sampled_aff = sampled_points_batch * binary_sampled_points_batch  # [Batch_size*pc_num, nsample, 3]

        else:
            sampled_points_batch = points
            sampled_points_aff_map_batch = points_aff_map
            binary_sampled_points_batch = binary_points_aff_map
            sampled_aff = points_aff

        points_features_list = []
        aff_features_list = []

        for i, knn_point in enumerate(self.knn_points):
            grouped_points = knn_groups(points,sampled_points_batch,knn_point) # [Batch_size*pc_num, points_num, k, 3]
            grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()  # [Batch_size*pc_num, points_num, k, C] -> [Batch_size*pc_num, C, points_num, k]
            grouped_aff = knn_groups_mask(points_aff, sampled_aff,binary_points_aff_map, knn_point) # [Batch_size*pc_num, points_num, k, 3]
            grouped_aff = grouped_aff.permute(0, 3, 1, 2).contiguous() # [Batch_size*pc_num, points_num, k, C] -> [Batch_size*pc_num, C, points_num, k]

            # 对点云进行编码
            grouped_points_features = grouped_points
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points_features = F.relu(bn(conv(grouped_points_features)), inplace=True)  # [Batch_size*pc_num, C', points_num, k]

                if j != len(self.conv_blocks[i]) - 1:
                    grouped_points_features = torch.cat((grouped_points_features, grouped_points),dim=1)  # [Batch_size*pc_num, C'+C, points_num, k]

            pooled_points_features = torch.max(grouped_points_features, dim=-1)[0]  # [Batch_size, C', points_num]
            points_features_list.append(pooled_points_features)

            # 对点云Aff进行编码
            grouped_aff_features = grouped_aff
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_aff_features = F.relu(bn(conv(grouped_aff_features,binary_sampled_points_batch),binary_sampled_points_batch), inplace=True)  # [Batch_size*pc_num, C', points_num, k]

                if j != len(self.conv_blocks[i]) - 1:
                    grouped_aff_features = torch.cat((grouped_aff_features, grouped_aff),dim=1)  # [Batch_size*pc_num, C'+C, points_num, k]

            pooled_aff_features = torch.max(grouped_aff_features, dim=-1)[0]  # [Batch_size, C', points_num]
            aff_features_list.append(pooled_aff_features)

        points_features = torch.cat(points_features_list, dim=1)  # [Batch_size*pc_num, sum(C'), points_num]
        points_features = points_features.permute(0, 2, 1).contiguous()  # [Batch_size*pc_num, points_num, sum(C')]
        points_features = self.embed_layer(points_features)  # [Batch_size*pc_num, points_num, embed_dim]
        points_features = points_features.view(Batch_size, pc_num, points_num,-1) # [Batch_size, pc_num, points_num, embed_dim]

        aff_features = torch.cat(aff_features_list, dim=1)  # [Batch_size*pc_num, sum(C'), points_num]
        aff_features = aff_features.permute(0, 2, 1).contiguous()  # [Batch_size*pc_num, points_num, sum(C')]
        aff_features = self.embed_layer(aff_features)  # [Batch_size*pc_num, points_num, embed_dim]
        aff_features = aff_features.view(Batch_size, pc_num, points_num,-1)  # [Batch_size, pc_num, points_num, embed_dim]
        aff_features = aff_features * binary_sampled_points_batch.view(Batch_size, pc_num, points_num, 1)

        sampled_points_aff_map_batch = sampled_points_aff_map_batch.view(Batch_size, pc_num, points_num, 1)

        return points_features, aff_features ,sampled_points_aff_map_batch



# 点云监督高斯进行语义层面粗学习
'''
    Input:
        __init__:
            gs_dim: int
            pc_dim: int
            embed_dim: int
            num_heads: int
            nsample: int
            knn_points: list
            mlp_list: list
            att_drop: float
            lin_drop: float
        forward:
            gs: [Batch_size, max_points_num, 10]
            gs_aff_map: [Batch_size, max_points_num, 1]
            pc: [Batch_size,pc_num,points_num,3]
            pc_aff_map: [Batch_size,pc_num,points_num,1]
            mask: [Batch_size, max_points_num]
            threshold: float
    Output:
        forward:
            gs_stru_features: [Batch_size, nsample, embed_dim]
            pc_stru_features: [B, pc_num, points_num, embed_dim]

'''
class CSA(nn.Module):
    def __init__(self,
                 gs_dim,
                 pc_dim,
                 embed_dim=256,
                 num_heads=4,
                 nsample=512,
                 knn_points=[4, 8, 16],
                 mlp_list=[[32, 32, 64], [64, 64, 125], [64, 64, 128]],
                 att_drop=0.,
                 lin_drop=0.):
        super(CSA, self).__init__()

        self.gs_encoder = gs_encoder(nsample,
                                     knn_points,
                                     gs_dim,
                                     mlp_list,
                                     embed_dim)

        self.pc_encoder = pc_encoder(nsample,
                                     knn_points,
                                     pc_dim,
                                     mlp_list,
                                     embed_dim)

        self.ca = Cross_MultiAttention_Q_masked(
            embed_dim,
            embed_dim,
            num_heads,
            dim_out=256,
            att_drop=att_drop,
            lin_drop=lin_drop,
            lin_before_qkv=False,
            lin_after_att=False)

        self.layernorm = nn.LayerNorm(embed_dim)

        self.ffn1 = FFN(embed_dim, [embed_dim])
        self.ffn2 = FFN(embed_dim, [embed_dim])

    def forward(self, gs, gs_aff_map, pc, pc_aff_map, pad_mask, threshold=0.5):
        # 计算点云的有效比例
        pc_binary_mask = (pc_aff_map > threshold).float()  # [B, pc_num, points_num, 1]
        pc_valid_ratio = pc_binary_mask.mean(dim=(-2, -1))  # [B, pc_num]
        max_pc_ratio = pc_valid_ratio.mean(dim=1, keepdim=True).view(-1, 1, 1)  # [B, 1, 1] 取每个batch的最大值

        gs_embed,gs_aff_embed,sampled_gs_aff_map = self.gs_encoder(gs, gs_aff_map, pad_mask, threshold,max_pc_ratio) # [Batch_size, nsample, embed_dim] ; [Batch_size, nsample, embed_dim]
        binary_gs_aff_map = Binary_mask(sampled_gs_aff_map,threshold=threshold) # [Batch_size, nsample, 1]
        gs_att_output, _ = self.ca(gs_aff_embed, gs_embed, gs_embed, sampled_gs_aff_map,binary_gs_aff_map)  # [Batch_size, nsample, embed_dim] # 非标注点为0
        gs_res = gs_att_output + gs_aff_embed  # [Batch_size, nsample, embed_dim]

        # 最大池化
        gs_res, _ = torch.max(gs_res, dim=1, keepdim=True)

        gs_stru_features =self.ffn1(gs_res)
        gs_stru_features = self.layernorm(gs_stru_features)
        gs_stru_features = self.ffn2(gs_stru_features) # [Batch_size, 1, embed_dim]

        pc_embed,pc_aff_embed, pc_aff_map = self.pc_encoder(pc, pc_aff_map, threshold) # [Batch_size, pc_num, points_num, embed_dim] ; [Batch_size, pc_num, points_num, embed_dim]
        Batch_size, pc_num, points_num, _ = pc_embed.shape
        pc_embed = pc_embed.view(Batch_size*pc_num,points_num,-1).contiguous() # [Batch_size*pc_num, points_num, embed_dim]
        pc_aff_embed = pc_aff_embed.view(Batch_size*pc_num,points_num,-1) # [Batch_size*pc_num, points_num, embed_dim]
        pc_aff_map = pc_aff_map.view(Batch_size*pc_num,points_num,-1) # [Batch_size*pc_num, points_num, 1]

        binary_pc_aff_map = Binary_mask(pc_aff_map,threshold=threshold)
        pc_att_output, _ = self.ca(pc_aff_embed, pc_embed, pc_embed,pc_aff_map,binary_pc_aff_map)  # [Batch_size*pc_num, points_num, embed_dim]
        pc_res = pc_att_output + pc_aff_embed  # [Batch_size*pc_num, points_num, embed_dim]

        # 最大池化
        pc_res, _ = torch.max(pc_res, dim=1, keepdim=True)

        pc_stru_features = self.ffn1(pc_res)
        pc_stru_features = self.layernorm(pc_stru_features)
        pc_stru_features = self.ffn2(pc_stru_features)  # [Batch_size, nsample, embed_dim] # 非标注点为0
        pc_stru_features = pc_stru_features.view(Batch_size, pc_num, 1, -1)  # [B, pc_num, 1, embed_dim]

        return gs_stru_features, pc_stru_features
