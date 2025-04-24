import torch
import torch.nn as nn
import torch.nn.functional as F

def standardize(tensor, dim=-1):
    mean = tensor.mean(dim=dim, keepdim=True)  # 计算均值
    std = tensor.std(dim=dim, keepdim=True)
    return (tensor - mean) / (std + 1e-8)

def cosine_loss(gs_features, pc_features, loss_consis_weight, device, neg_ratio=0):
    B, _, C = gs_features.size()

    gs_norm = F.normalize(gs_features.squeeze(1), p=2, dim=-1)
    pc_norm = F.normalize(pc_features.view(B, -1, C), p=2, dim=-1)

    pc_num = pc_norm.shape[1]
    
    total_loss = 0.0
    for i in range(pc_num):
        pc_i = pc_norm[:, i, :]  # [B, C]
        
        dot_product = (gs_norm * pc_i).sum(dim=-1)  # [B]
        
        loss_i = 1 - dot_product  # [B]
        loss_i = loss_i * loss_consis_weight[:,i]
        total_loss += loss_i.mean()
        
    avg_loss = total_loss
    return avg_loss




def pred_map_regularizer(pred_aff_map,
                         mask_batch,
                         epoch=None,
                         total_epoch=None,
                         epsilon=1e-8,
                         target_center=0.5,
                         boundary_margin=0.5):


    sum_aff = pred_aff_map.sum(dim=(-1, -2))  # [B]
    sum_mask = mask_batch.sum(dim=(-1, -2)) + epsilon  # [B]
    mean_aff = sum_aff / sum_mask  # [B]


    if epoch is not None and total_epoch is not None:
        target = 0.4 + 0.4 * (epoch / total_epoch)
    else:
        target = target_center
    center_reg = ((mean_aff - target) ** 2).mean()


    lower_penalty = torch.exp(10 * (0.01 - mean_aff))
    upper_penalty = torch.exp(10 * (mean_aff - 0.99))
    boundary_reg = (lower_penalty + upper_penalty).mean()

    boundary_weight = boundary_margin
    center_weight = 1 - boundary_margin

    return boundary_weight * boundary_reg + center_weight * center_reg



'''
    Input:
        pred: [Batch_size, batch_max_points, 1]
        target: [Batch_size, batch_max_points, 1]
        mask: [Batch_size, batch_max_points]
    Output:
        dice_loss: float
'''
def Dice_loss(pred, target, mask):
    mask = mask.unsqueeze(-1)
    intersection_positive = torch.sum(pred * target * mask, 1) # [Batch_size, 1, 1]
    cardinality_positive = torch.sum(pred * mask, 1) + torch.sum(target * mask, 1) # [Batch_size, 1]
    dice_positive = (2. * intersection_positive + 1e-6) / (cardinality_positive + 1e-6) # [Batch_size, 1]

    intersection_negative = torch.sum((1. - pred) * (1. - target) * mask, 1) # [Batch_size, 1]
    cardinality_negative = torch.sum((1 - pred) * mask, 1) + torch.sum((1 - target) * mask, 1) # [Batch_size, 1]
    dice_negative = (2. * intersection_negative + 1e-6) / (cardinality_negative + 1e-6) # [Batch_size, 1]

    dice_loss = torch.sum(2 - dice_positive - dice_negative)

    return dice_loss



'''
    Input:
        pred: [Batch_size, batch_max_points, 1]
        target: [Batch_size, batch_max_points, 1]
        mask: [Batch_size, batch_max_points]
        alpha: float
        gamma: int
    Output:
        ce_loss: float
'''
def CE_loss(pred, target, mask , alpha=0.25, gamma=2):
    negative_loss = -(1 - alpha) * torch.mul(pred ** gamma, torch.mul(1 - target, torch.log(1 - pred + 1e-6))) # [Batch_size, batch_max_points, 1]
    positive_loss = -alpha * torch.mul((1 - pred) ** gamma, torch.mul(target, torch.log(pred + 1e-6))) # [Batch_size, batch_max_points, 1]

    mask = mask.unsqueeze(-1) # [Batch_size, batch_max_points, 1]
    ce_loss = negative_loss + positive_loss # [Batch_size, batch_max_points, 1]
    ce_loss = ce_loss * mask # [Batch_size, batch_max_points, 1]

    num_valid = mask.sum(dim=(-1,-2)) + 1e-6 # [Batch_size]
    ce_loss = torch.sum(ce_loss.sum(dim=(-1,-2)) / num_valid) # float
    return ce_loss


