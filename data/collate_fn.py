# -*- coding: UTF-8 -*-
import torch.utils.data as data
import os
import random
import open3d as o3d
import json
import sys
import plyfile
import numpy as np
import pandas as pd
import re
import torch
from scipy.spatial import KDTree
from models.Pointnet_utils import fps_one


# 用0填充高斯模型，每个batch的高斯模型填充至该批次的最大点数
def pad_gs(gs,max_points):
    gs_points, C = gs.shape
    if gs_points != max_points:
        need_fill_points = max_points - gs_points
        fill_data = np.zeros((need_fill_points, C))
        padded_gs = np.concatenate((gs, fill_data), axis=0)
        mask = np.concatenate((np.ones(gs_points), np.zeros(need_fill_points)), axis=0)
    else:
        padded_gs=gs
        mask = np.ones(max_points)
    return padded_gs, mask

# 重写数据裁剪函数，将每个batch的gs数据采样成改batch下的最低点数
def collate_fn_train(batch):
    # 提取其他数据
    obj_batch = [item[0] for item in batch]
    aff_batch = [item[1] for item in batch]
    pc_mean_all_batch = [item[4] for item in batch]
    pc_aff_map_all_batch = [item[5] for item in batch]
    question_batch = [item[6] for item in batch]
    answer_batch = [item[7] for item in batch]

    # 提取批次中的高斯数据并采样成相同尺寸
    gs_datas=[data[2] for data in batch]
    min_points = min(gs.shape[0] for gs in gs_datas)
    max_points = max(gs.shape[0] for gs in gs_datas)
    # 快速点采样(由于每个高斯模型的元素数不一致，因此需要采样，使得数据尺寸一致（如此才等使用dataloader批量加载数据）)
    sampled_gs_datas = [fps_one(gs, min_points) for gs in gs_datas]

    # 填充高斯模型
    padded_gs_datas=[]
    mask_batch=[]
    for gs in gs_datas:
        padded_gs, mask = pad_gs(gs, max_points)
        padded_gs_datas.append(padded_gs)
        mask_batch.append(mask)

    # 得到高斯标注的掩码
    gs_aff_maps = [data[3] for data in batch]
    if not any(x is None for x in gs_aff_maps):
        gs_aff_maps = [np.expand_dims(data, axis=-1) for data in gs_aff_maps]
        padded_gs_aff_maps=[]
        # 填充高斯掩码
        for gs_aff_map in gs_aff_maps:
            padded_gs_aff_map, _ = pad_gs(gs_aff_map, max_points)
            padded_gs_aff_maps.append(padded_gs_aff_map)  # [Batch size, batch_max_points, 1]
        return obj_batch, aff_batch, sampled_gs_datas, padded_gs_datas, mask_batch, padded_gs_aff_maps, pc_mean_all_batch, pc_aff_map_all_batch, question_batch, answer_batch

    return obj_batch, aff_batch, sampled_gs_datas, padded_gs_datas, mask_batch, gs_aff_maps, pc_mean_all_batch, pc_aff_map_all_batch, question_batch, answer_batch



def collate_fn_val_test(batch):
    # 提取其他数据
    obj_batch = [item[0] for item in batch]
    aff_batch = [item[1] for item in batch]
    questions_batch = [item[4] for item in batch]
    answer_batch = [item[5] for item in batch]
    gs_file_path_batch = [item[6] for item in batch]

    # 提取批次中的高斯数据并采样成相同尺寸
    gs_datas=[data[2] for data in batch]
    min_points = min(gs.shape[0] for gs in gs_datas)
    max_points = max(gs.shape[0] for gs in gs_datas)
    # 快速点采样(由于每个高斯模型的元素数不一致，因此需要采样，使得数据尺寸一致（如此才能使用dataloader批量加载数据）)
    sampled_gs_datas = [fps_one(gs, min_points) for gs in gs_datas]

    # 填充高斯模型
    padded_gs_datas=[]
    mask_batch=[]
    for gs in gs_datas:
        padded_gs, mask = pad_gs(gs, max_points)
        padded_gs_datas.append(padded_gs)
        mask_batch.append(mask)

    # 高斯标注的掩码
    gs_aff_maps = [data[3] for data in batch]
    gs_aff_maps = [np.expand_dims(data,axis=-1) for data in gs_aff_maps]

    padded_gs_aff_maps = []
    # 填充高斯掩码
    for gs_aff_map in gs_aff_maps:
        padded_gs_aff_map, _ = pad_gs(gs_aff_map, max_points)
        padded_gs_aff_maps.append(padded_gs_aff_map)
    return obj_batch, aff_batch, sampled_gs_datas, padded_gs_datas, mask_batch, padded_gs_aff_maps, questions_batch, answer_batch, gs_file_path_batch

