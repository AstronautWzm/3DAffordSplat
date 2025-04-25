# -*- coding: UTF-8 -*-
import torch.utils.data as data
import os
import random
import open3d as o3d
import json
import sys
import os
import plyfile
import numpy as np
import pandas as pd
import re
import torch
from scipy.spatial import KDTree
from models.Pointnet_utils import fps_one


# 将项目文件夹添加到系统路径中
project_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir_path not in sys.path:
    sys.path.append(project_dir_path)


# 数据进行归一化处理，使其质心位于原点，并且所有点的坐标范围归一化到单位球内
def mean_normalize(points):
    centroid=np.mean(points, axis=0)
    points=points-centroid
    distance=np.max(np.sqrt(np.sum(points**2,axis=1)))
    points=points/distance
    return points, centroid, distance

# 得到one-hot掩码
def get_gs_aff_map(gs_xyz,aff_xyz):
    # 构建 KDTree
    kdtree = KDTree(gs_xyz)
    distances, indices = kdtree.query(aff_xyz, k=1)
    aff_map = np.zeros(gs_xyz.shape[0], dtype=int)
    aff_map[indices] = 1
    return aff_map



# 训练集
'''
    Input:
        __init__:
            pc_num: int
            setting: Seen or UnSeen
            is_pretrain:bool
            gs_label_num: int
            random_seed: int
            root_dir: str
    Output:
        __getitem__:
            obj: [Batch size]
            aff: [Batch size]
            gs_features: [Batch size, batch_min_points, 10]
            padded_gs_datas: [Batch size, batch_max_points, 10]
            masks: [Batch size, batch_max_points]
            gs_aff_maps: [Batch size](None) or [Batch size, batch_max_points, 1]
            pc_mean_all: [Batch size, pc_num, pc_points(2048), 3]
            pc_aff_map_all:[Batch size, pc_num, pc_points(2048), 1]
            question: [Batch size]
            answer: [Batch size]
'''
class QAffordSplat_train(data.Dataset):
    def __init__(self,pc_num=5,setting='Seen',is_pretrain=True,gs_label_num='all',random_seed=0,root_dir='AffordSplat'):
        self.setting=setting
        self.is_pretrain=is_pretrain
        self.gs_label_num=gs_label_num

        # 抽取pc文件的随机种子
        random.seed(random_seed)

        # 文件路径
        self.text_path = os.path.join(project_dir_path,root_dir+'/Affordance-Question.csv')
        self.root_path = os.path.join(project_dir_path,root_dir+'/Seen/train')
        self.file_stru_path = os.path.join(project_dir_path,root_dir+'/obj_aff_structure.json')
        self.UnSeen_train_path = os.path.join(project_dir_path,root_dir+'/UnSeen_train.json')

        if setting == 'UnSeen':
            # 加载UnSeen配置
            with open(self.UnSeen_train_path,'r') as file:
                UnSeen_train = json.load(file)

        # 加载文本数据
        with open(self.text_path, 'r') as file:
            self.text_data=pd.read_csv(file)

        # 加载训练数据结构
        with open(self.file_stru_path, 'r') as f:
            file_stru=json.load(f)

        # 加载gs、pc训练数据列表
        self.gs_files=[]
        self.pc_files=[]

        if self.setting == 'UnSeen':
            file_stru = UnSeen_train
        elif self.setting == 'Seen':
            pass
        else:
            raise ValueError('setting must be Seen or UnSeen')

        # 加载gs、pc训练数据文件列表
        if self.is_pretrain: # 如果是预训练模型，则不加载高斯标注，直接加载原文件列表
            for obj in file_stru:
                for aff in file_stru[obj]:
                    pc_files_all=[file.replace('_anno','').replace('.json','.ply') for file in os.listdir(os.path.join(self.root_path,obj,aff)) if file.endswith('.json')]
                    for gs in os.listdir(os.path.join(self.root_path,obj,'Gaussian')):
                        gs_file=obj+'/'+aff+'/'+gs
                        self.gs_files.append(gs_file)

                        pc_file=random.sample(pc_files_all,pc_num)
                        self.pc_files.append(pc_file)

        else: # 如果是微调模型，则加载高斯标注，根据标注文件反溯原文件
            if self.gs_label_num is None:
                raise ValueError('When you set is_pretrain=False, gs_label_num must be set')

            for obj in file_stru:
                for aff in file_stru[obj]:
                    pc_files_all = [file.replace('_anno', '').replace('.json', '.ply') for file in
                                    os.listdir(os.path.join(self.root_path, obj, aff)) if file.endswith('.json')]
                    gs_files_all = [obj + '/' + aff + '/' + file.replace('_anno', '') for file in
                                    os.listdir(os.path.join(self.root_path, obj, aff)) if file.endswith('.ply')]

                    if self.gs_label_num !='all': # 每个 obj-aff 抽取最多 gs_label_num 个数据
                        if len(self.gs_files) > int(self.gs_label_num):
                            gs_files_all = random.sample(gs_files_all, int(self.gs_label_num))

                    for i in range(len(gs_files_all)):
                        pc_file = random.sample(pc_files_all, pc_num)
                        self.pc_files.append(pc_file)
                        self.gs_files.append(gs_files_all[i])

    def __len__(self):
        return len(self.gs_files)

    def read_gs(self,path):
        gs_data=plyfile.PlyData.read(path)['vertex']

        # 读取高斯的均值 [x,y,z]
        gs_x=list(gs_data['x'])
        gs_y=list(gs_data['y'])
        gs_z=list(gs_data['z'])
        gs_mean=np.column_stack((gs_x,gs_y,gs_z))
        gs_mean_norm,_,_=mean_normalize(gs_mean)

        # 读取高斯的方差 [scale,rotation]
        gs_scale_0=list(gs_data['scale_0'])
        gs_scale_1=list(gs_data['scale_1'])
        gs_scale_2=list(gs_data['scale_2'])
        gs_scale=np.column_stack((gs_scale_0,gs_scale_1,gs_scale_2))
        gs_rot_0=list(gs_data['rot_0'])
        gs_rot_1=list(gs_data['rot_1'])
        gs_rot_2=list(gs_data['rot_2'])
        gs_rot_3=list(gs_data['rot_3'])
        gs_rot=np.column_stack((gs_rot_0,gs_rot_1,gs_rot_2,gs_rot_3))

        # 将gs的均值特征和方差特征组合在一起
        gs_features=np.column_stack((gs_mean_norm,gs_scale,gs_rot))

        return gs_features, gs_mean

    def __getitem__(self, index):
        # 读取obj、aff类别
        gs_file = self.gs_files[index]
        obj=gs_file.split('/')[0]
        aff=gs_file.split('/')[1]
        gs_file=gs_file.split('/')[-1]

        # 读取gs文件
        gs_file_path = os.path.join(self.root_path,obj,'Gaussian',gs_file)
        gs_features,gs_mean = self.read_gs(gs_file_path)
        gs_aff_map = None

        # 读取gs标签
        if not self.is_pretrain:
            gs_file = gs_file.replace('GS_', 'GS_anno_')
            gs_label_path = os.path.join(self.root_path,obj,aff,gs_file)
            _,gs_label = self.read_gs(gs_label_path)
            gs_xyz = gs_mean[:, :3]
            aff_xyz = gs_label[:, :3]
            gs_aff_map=get_gs_aff_map(gs_xyz, aff_xyz)


        # 读取pc文件和 aff map
        pc_mean_all=[]
        pc_aff_map_all=[]
        pc_files=self.pc_files[index]
        for pc_file in pc_files:
            pc_file=os.path.join(self.root_path,obj,'PointCloud',pc_file)
            pc_data=plyfile.PlyData.read(pc_file)['vertex']
            pc_x=list(pc_data['x'])
            pc_y=list(pc_data['y'])
            pc_z=list(pc_data['z'])
            pc_mean=np.column_stack((pc_x,pc_y,pc_z))
            pc_mean,_,_=mean_normalize(pc_mean)
            pc_mean_all.append(pc_mean)

            pc_id = re.search(r"PC_(\d+)\.ply", pc_file).group(1)
            pc_aff_file=os.path.join(self.root_path,obj,aff,"PC_anno_"+pc_id+".json")
            with open(pc_aff_file, 'r') as file:
                pc_aff_map=np.array(json.load(file))
                pc_aff_map = np.expand_dims(pc_aff_map,axis=-1) # [npoints,1]
            pc_aff_map_all.append(pc_aff_map)
        pc_mean_all = np.array(pc_mean_all)
        pc_aff_map_all=np.array(pc_aff_map_all)

        # 读取文本问题和回答
        row=self.text_data[(self.text_data['Object'] == obj) & (self.text_data['Affordance'] == aff)]
        question = str(row[random.choice([f'Question{i}' for i in range(15)])].values[0])
        answer=str(row['Answer2'].values[0])

        return obj,aff,gs_features,gs_aff_map,pc_mean_all,pc_aff_map_all,question,answer



# 验证集
'''
    Input:
        __init__:
            setting: Seen or UnSeen
            state: val or test
            root_dir: str
    Output:
        __getitem__:
            obj: [Batch size]
            aff: [Batch size]
            gs_features: [Batsh size, batch_min_points, 10]
            padded_gs_datas: [Batsh size, batch_max_points, 10]
            masks: [Batsh size, batch_max_points]
            gs_aff_maps: [Batch size, batch_max_points, 1]
            question: [Batch size,15]
            answer: [Batch size]
'''
class QAffordSplat_val_test(data.Dataset):
    def __init__(self,setting='Seen',state='val',root_dir='AffordSplat'):
        self.setting=setting
        # 文件路径
        if state == 'val' or state == 'test':
            self.root_path = os.path.join(project_dir_path,root_dir+'/Seen/'+state)
        else:
            raise ValueError('state must be either val or test')

        self.file_stru_path = os.path.join(project_dir_path, root_dir+'/obj_aff_structure.json')
        self.UnSeen_val_path = os.path.join(project_dir_path, root_dir+'/UnSeen_test.json')
        self.text_path = os.path.join(project_dir_path, root_dir+'/Affordance-Question.csv')

        if setting == 'UnSeen':
            # 加载UnSeen配置
            with open(self.UnSeen_val_path,'r') as file:
                UnSeen_val = json.load(file)

        # 加载文本数据
        with open(self.text_path, 'r') as file:
            self.text_data=pd.read_csv(file)

        # 加载数据类型结构
        with open(self.file_stru_path, 'r') as f:
            file_stru=json.load(f)

        # 加载gs训练数据列表
        self.gs_files = []

        if self.setting == 'UnSeen':
            file_stru = UnSeen_val
        elif self.setting == 'Seen':
            pass
        else:
            raise ValueError('setting must be Seen or UnSeen')

        for obj in file_stru:
            for aff in file_stru[obj]:
                gs_files_all = [obj + '/' + aff + '/' + file.replace('_anno', '') for file in
                                os.listdir(os.path.join(self.root_path, obj, aff)) if file.endswith('.ply')]

                for i in range(len(gs_files_all)):
                    self.gs_files.append(gs_files_all[i])

    def __len__(self):
        return len(self.gs_files)

    def read_gs(self,path):
        gs_data=plyfile.PlyData.read(path)['vertex']

        # 读取高斯的均值 [x,y,z]
        gs_x=list(gs_data['x'])
        gs_y=list(gs_data['y'])
        gs_z=list(gs_data['z'])
        gs_mean=np.column_stack((gs_x,gs_y,gs_z))
        gs_mean_norm,_,_=mean_normalize(gs_mean)

        # 读取高斯的方差 [scale,rotation]
        gs_scale_0=list(gs_data['scale_0'])
        gs_scale_1=list(gs_data['scale_1'])
        gs_scale_2=list(gs_data['scale_2'])
        gs_scale=np.column_stack((gs_scale_0,gs_scale_1,gs_scale_2))
        gs_rot_0=list(gs_data['rot_0'])
        gs_rot_1=list(gs_data['rot_1'])
        gs_rot_2=list(gs_data['rot_2'])
        gs_rot_3=list(gs_data['rot_3'])
        gs_rot=np.column_stack((gs_rot_0,gs_rot_1,gs_rot_2,gs_rot_3))

        # 将gs的均值特征和方差特征组合在一起
        gs_features=np.column_stack((gs_mean_norm,gs_scale,gs_rot))

        return gs_features, gs_mean

    def __getitem__(self, index):

        # 读取obj、aff类别
        gs_file = self.gs_files[index]
        obj = gs_file.split('/')[0]
        aff = gs_file.split('/')[1]
        gs_file = gs_file.split('/')[-1]

        # 读取gs文件
        gs_file_path = os.path.join(self.root_path, obj, 'Gaussian', gs_file)
        gs_features,gs_mean = self.read_gs(gs_file_path)

        # 读取高斯标签
        gs_file = gs_file.replace('GS_', 'GS_anno_')
        gs_label_path = os.path.join(self.root_path, obj, aff, gs_file)
        _,gs_label = self.read_gs(gs_label_path)

        gs_xyz = gs_mean[:, :3]
        aff_xyz = gs_label[:, :3]
        gs_aff_map = get_gs_aff_map(gs_xyz, aff_xyz)

        # 读取文本问题和回答
        row=self.text_data[(self.text_data['Object'] == obj) & (self.text_data['Affordance'] == aff)]
        questions = str(row[[f'Question{i}' for i in range(15)]].values[0])
        answer=str(row['Answer2'].values[0])

        return obj,aff,gs_features,gs_aff_map,questions,answer,gs_file_path




