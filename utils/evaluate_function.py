import numpy as np
import torch
from sklearn.metrics import roc_auc_score

eps = 1e-12



'''
    Input:
        pred: [Batch_size, batch_max_points, 1]
        target: [Batch size, batch_max_points, 1]
        mask: [Batsh size, batch_max_points]
    Output:
        total_error: [Batch size] 
        num_valid_points: [Batch size] 
'''
def MAE(pred,target,mask):
    mask = mask.unsqueeze(-1)  # [Batsh size, batch_max_points， 1]

    error = torch.abs(pred* mask - target* mask)
    masked_error = error
    total_error = torch.sum(masked_error,dim=(1,2)) # [Batch size]
    num_valid_points = torch.sum(mask,dim=(1,2)) # [Batch size]

    return total_error, num_valid_points



'''
    Input:
        pred: [Batch_size, batch_max_points, 1]
        target: [Batch size, batch_max_points, 1]
        mask: [Batsh size, batch_max_points]
    Output:
        kld: [Batsh size]
'''
def KLD(pred,target,mask):
    mask = mask.unsqueeze(-1)

    pred = pred * mask
    target = target * mask

    pred_sum = torch.sum(pred, dim=(1, 2), keepdim=True) + eps
    target_sum = torch.sum(target, dim=(1, 2), keepdim=True) + eps
    pred_norm = pred / pred_sum
    target_norm = target / target_sum

    kld = torch.sum(target_norm * torch.log((target_norm + eps) / (pred_norm + eps)), dim=(1, 2))

    return kld



'''
    Input:
        pred: [Batch_size, batch_max_points, 1]
        target: [Batch size, batch_max_points, 1]
        mask: [Batsh size, batch_max_points]
    Output:
        sim: [Batsh size]
'''
def SIM(pred,target,mask):
    mask = mask.unsqueeze(-1)

    pred = pred * mask
    target = target * mask

    pred_sum = torch.sum(pred, dim=(1, 2), keepdim=True) + eps
    target_sum = torch.sum(target, dim=(1, 2), keepdim=True) + eps

    pred_norm = pred / pred_sum
    target_norm = target / target_sum

    intersection = torch.minimum(pred_norm, target_norm)

    sim = torch.sum(intersection, dim=(1, 2))
    return sim



'''
    Input:
        pred: [Batch_size, batch_max_points, 1]
        target: [Batch size, batch_max_points, 1]
        mask: [Batsh size, batch_max_points]
    Output:
        auc_tensor: [Batsh size]
'''
def AUC(pred,target,mask):
    batch_size = pred.shape[0]
    auc_values = []

    for i in range(batch_size):
        bool_mask = mask[i] == 1
        valid_preds = pred[i, bool_mask, 0].detach().cpu().numpy() # [N]
        valid_targets = target[i, bool_mask, 0].detach().cpu().numpy() # [N]

        if np.sum(valid_targets) == 0 or np.sum(1 - valid_targets) == 0:
            auc = float('nan')
        else:
            auc = roc_auc_score(valid_targets, valid_preds)

        auc_values.append(auc)

    auc_tensor = torch.tensor(auc_values, device=pred.device)
    return auc_tensor



'''
    Input:
        pred: [Batch_size, batch_max_points, 1]
        target: [Batch size, batch_max_points, 1]
        mask: [Batsh size, batch_max_points]
    Output:
        iou_tensor: [Batsh size]
'''
def IOU(pred, target, mask):
    batch_size = pred.shape[0]
    iou_values = []
    thresholds = np.linspace(0, 1, 20)

    for i in range(batch_size):
        bool_mask = mask[i].bool()
        valid_preds = pred[i, bool_mask, 0].cpu().numpy().ravel()
        valid_targets = target[i, bool_mask, 0].cpu().numpy().ravel().astype(int)

        # 全零目标过滤
        if len(valid_targets) == 0 or np.all(valid_targets == 0):
            iou_values.append(float('nan'))
            continue

        temp_iou = []
        for thres in thresholds:
            pred_mask = (valid_preds >= thres).astype(int)
            intersection = np.logical_and(pred_mask, valid_targets).sum()
            union = np.logical_or(pred_mask, valid_targets).sum()

            if union > 0:
                temp_iou.append(intersection / union)
            else:
                temp_iou.append(float('nan'))


        valid_iou = np.nanmean(np.array(temp_iou))
        iou_values.append(valid_iou if not np.isnan(valid_iou) else float('nan'))

    return torch.tensor(iou_values, device=pred.device, dtype=torch.float32)



