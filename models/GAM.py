import torch
import torch.nn as nn
import torch.nn.functional as F

# 上采样回原点数(Inverse Distance Weighted Interpolation)
'''
    Input:
        __init__:
             k: int
             p: int
        forward:
            xyz: [Batch_size, batch_max_points, 3]
            sampled_xyz: [Batch_size, nsamples, 3]
            features: [Batch_size, batch_max_points, C]
            sampled_features: [Batch_size, nsamples, C']
    Output:
        forward:
            [Batch_size, batch_max_points, C+C']
'''
class Optimized_Upsample(nn.Module):
    def __init__(self, k=3, p=2):
        super().__init__()
        self.k = k
        self.p = p

    def forward(self, xyz, sampled_xyz, features, sampled_features, masks):
        Batch_size, batch_max_points, _ = xyz.shape
        output = []

        # 逐样本处理
        for i in range(Batch_size):
            # 提取有效点
            valid_mask = masks[i].bool()  # [N]
            valid_xyz = xyz[i][valid_mask]  # [valid_points,3]
            valid_features = features[i][valid_mask] if features else None  # [valid_points,C]

            # 仅计算有效点的插值
            dists = self._pairwise_distance(valid_xyz, sampled_xyz[i])  # [valid_points,nsamples]

            # 找k近邻并计算权重
            min_dists, idx = dists.topk(self.k, dim=-1, largest=False)  # [valid_points,k]
            min_dists = min_dists.clamp(min=1e-10)
            weights = 1.0 / (min_dists  ** self.p + 1e-10)  # [valid_points,k]
            weights /= weights.sum(dim=-1, keepdim=True)

            # 收集特征并加权
            gathered = sampled_features[i][idx]  # [valid_points,k,C']
            interpolated = (gathered * weights.unsqueeze(-1)).sum(dim=1)  # [n_valid,C']

            # 特征拼接
            if valid_features is not None:
                combined = torch.cat([interpolated, valid_features], dim=-1)  # [n_valid,C+C']
            else:
                combined = interpolated # [n_valid,C']

            # 填充到原始长度
            padded = torch.zeros(batch_max_points, combined.shape[-1], device=xyz.device)
            padded[valid_mask] = combined
            output.append(padded)

        return torch.stack(output, dim=0)  # [B,N,C+C']

    # 分块计算避免OOM
    '''
    Input:
        valid_xyz: [valid_points,3]
        sampled_xyz: [nsamples, 3]
    Output:
        dists: [valid_points,nsamples]
    '''
    @staticmethod
    def _pairwise_distance(valid_xyz, sampled_xyz):
        chunk_size = 512  # 根据显存调整
        dists = []
        for i in range(0, valid_xyz.shape[0], chunk_size):
            chunk = valid_xyz[i:i + chunk_size]  # [chunk,3]
            delta = chunk.unsqueeze(1) - sampled_xyz.unsqueeze(0)  # [chunk,nsamples,3]
            chunk_dists = (delta ** 2).sum(dim=-1)  # [chunk,nsamples]
            dists.append(chunk_dists)
        return torch.cat(dists, dim=0)  # [valid_points,nsamples]



# 上采样方法(Inverse Distance Weighted Interpolation)
'''
    Input:
        __init__:
             k: int
             p: int
             block_size: int
        _blockwise_knn:
             xyz: [Batch_size, nsamples, 3]
             sampled_xyz: [Batch_size, nsamples', 3]
        forward:
            xyz: [Batch_size, nsamples, 3]
            sampled_xyz: [Batch_size, nsamples', 3]
            features: [Batch_size, nsamples, C]
            sampled_features: [Batch_size, nsamples', C']
    Output:
        _blockwise_knn:
            idx: [Batch_size, nsamples, k]
            dists: [Batch_size, nsamples, k]
        forward:
            [Batch_size, nsamples, C+C']
'''
class Upsample(nn.Module):
    def __init__(self, k=3, p=2, block_size=512):
        super(Upsample, self).__init__()
        self.k = k
        self.p = p
        self.block_size = block_size  # 根据GPU显存调整分块大小

    def forward(self, xyz, sampled_xyz, features, sampled_features):
        if xyz.size(1) == sampled_xyz.size(1):
            return sampled_features

        # 分块计算KNN索引和距离
        idx, dists = self._blockwise_knn(xyz, sampled_xyz, self.k)

        dists = dists.clamp(min=1e-10)
        weights = 1.0 / (dists ** self.p+ 1e-7)
        weights = weights / (weights.sum(dim=-1, keepdim=True)+ 1e-10)

        expanded_idx = idx.unsqueeze(-1).expand(-1, -1, -1, sampled_features.shape[-1])
        interpolated_features = torch.gather(
            sampled_features.unsqueeze(1).expand(-1, xyz.size(1), -1, -1),
            dim=2,
            index=expanded_idx
        )
        interpolated_features = (interpolated_features * weights.unsqueeze(-1)).sum(dim=2)

        if features is not None:
            upsampled_points = torch.cat([interpolated_features, features], dim=-1)
        else:
            upsampled_points = interpolated_features

        return upsampled_points

    # 分块计算KNN，避免一次性生成全距离矩阵
    def _blockwise_knn(self, xyz, sampled_xyz, k):

        B, N, _ = xyz.shape
        M = sampled_xyz.shape[1]
        device = xyz.device

        # 初始化结果张量
        final_dists = torch.full((B, N, k), float('inf'), device=device)
        final_idx = torch.zeros((B, N, k), dtype=torch.long, device=device)

        # 分块处理sampled_xyz
        for i in range(0, M, self.block_size):
            block_sampled = sampled_xyz[:, i:i + self.block_size, :]  # [B, block, 3]

            # 计算分块距离矩阵
            dist_block = torch.sum(xyz ** 2, dim=-1, keepdim=True)  # [B, N, 1]
            dist_block = dist_block - 2 * torch.einsum('bnd,bmd->bnm', xyz, block_sampled)
            dist_block += torch.sum(block_sampled ** 2, dim=-1).unsqueeze(1)  # [B, N, block]

            # 更新当前分块的topk
            current_topk_values, current_topk_idx = torch.topk(dist_block, k=k, dim=-1, largest=False)
            current_topk_idx = current_topk_idx + i  # 修正索引为全局索引

            # 合并结果
            merge_values = torch.cat([final_dists, current_topk_values], dim=-1)
            merge_idx = torch.cat([final_idx, current_topk_idx], dim=-1)

            # 保留全局最小的k个
            new_topk_values, new_topk_indices = torch.topk(merge_values, k=k, dim=-1, largest=False)
            final_dists = torch.gather(merge_values, -1, new_topk_indices)
            final_idx = torch.gather(merge_idx, -1, new_topk_indices)

        return final_idx, final_dists


# 粒度门控矩阵（低秩分解）
class DynamicLowRank(nn.Module):
    def __init__(self, input_dim, output_dim, rank=16):
        super().__init__()

        # 初始化低秩矩阵
        self.U = nn.Linear(input_dim, rank, bias=False)  # 降维
        self.V = nn.Linear(rank, output_dim, bias=False)  # 升维

        nn.init.orthogonal_(self.U.weight)
        nn.init.kaiming_uniform_(self.V.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        x = F.layer_norm(x, (x.size(-1),))
        return self.V(F.gelu(self.U(x)))


# 粒度自适应操作
'''
    Input:
        __init__:
             nsamples_list: [nsample_1, nsample_2, nsample_3,......]
             in_channels: int
             out_channels: int
             mlp_ratio: int
        forward:
            features_list: [[Batch_size, nsample_3, in_channels], [Batch_size, nsample_2, in_channels], ......]
            xyz_list: [[Batch_size, nsample_3, 3], [Batch_size, nsample_2, 3], ......]
    Output:
        forward:
            att_output: [Batch_size, nsamples_list[0], out_channels]
            weights: [Batch_size, num_granularity, out_channels]
'''
class GAM(nn.Module):
    def __init__(self,nsamples_list,in_channels,out_channels=None,mlp_ratio=2,k=3, p=2):
        super(GAM, self).__init__()
        self.nsamples_list = nsamples_list
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = Upsample(k,p)

        if out_channels:
            self.proj_list = nn.ModuleList()
            for i in range(len(self.nsamples_list)):
                self.proj_list.append(nn.Sequential(
                    nn.Linear(self.in_channels, self.in_channels * mlp_ratio),
                    nn.ReLU(),
                    nn.Linear(self.in_channels * mlp_ratio, self.out_channels)
                ))
            self.weights_gate = DynamicLowRank(nsamples_list[0] * out_channels, out_channels) # [nsamples_list[0]*out_channels,out_channels]
        else:
            self.weights_gate = DynamicLowRank(nsamples_list[0] * in_channels, in_channels) # [nsamples_list[0]*in_channels,in_channels]

    def forward(self,features_list,xyz_list):
        proj_features_list = []
        if self.out_channels:
            C=self.out_channels
        else:
            C=self.in_channels

        # 上采样到统一点数，[Batch_size, nsamples_list[0], sum(C)]
        for i in range(len(features_list)):
            sampled_xyz=xyz_list[i]
            sampled_features = features_list[i]
            upsampled_features = self.upsample(xyz_list[-1], sampled_xyz, None, sampled_features) # [Batch_size, nsamples = nsamples_list[0], in_channels]

            # 变化到统一维度
            if self.out_channels:
                upsampled_features = self.proj_list[i](upsampled_features) # [Batch_size, nsamples, out_channels]

            proj_features_list.append(upsampled_features) # [nfeatures, Batch_size, nsamples, out_channels]

        concated_features = torch.stack(proj_features_list, dim=1)  # [Batch_size, num_granularity, nsamples, out_channels]
        concated_features = concated_features.view(concated_features.size(0), concated_features.size(1), -1)  # [Batch_size, num_granularity, nsamples*out_channels]

        output_weights = F.softmax(self.weights_gate(concated_features), dim=1) # [Batch_size, num_granularity, out_channels]

        weights = output_weights.unsqueeze(2)  # [Batch_size, num_granularity, 1, out_channels]
        concated_features = concated_features.reshape(concated_features.size(0), concated_features.size(1),self.nsamples_list[0],C)  # [Batch_size, num_granularity, nsamples, out_channels]
        fused_features = weights * concated_features  # [Batch_size, num_granularity, nsamples, out_channels]
        fused_features = torch.sum(fused_features, dim=1)  # [Batch_size, nsamples, out_channels]

        return fused_features, output_weights
