import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement
import time
import yaml
import copy
from tqdm import tqdm
import shutil
from sklearn.metrics import *
from collections import defaultdict

# 将项目文件夹添加到系统路径中
project_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir_path not in sys.path:
    sys.path.append(project_dir_path)

import torch.distributed as dist

from data.dataloader import *
from data.collate_fn import *
from utils.logger import logger_init
from utils.loss_function import *
from models.Affordsplat_net import *
from utils.evaluate_function import *


def config_read(path):
    with open(path, 'r', encoding='utf-8') as f:
        args = yaml.safe_load(f)
    return args


class Get_ply_result():
    def __init__(self,apply_threshold):

        self.apply_threshold = apply_threshold

        # 读取配置
        self.model_args = config_read('config/model_config.yaml')
        self.data_args = config_read('config/data_config.yaml')
        self.train_args = config_read('config/train_config.yaml')

        self.inference_args = self.train_args['inference']

        self.ckpt_path = self.inference_args['ckpt_path']
        self.setting = self.data_args['setting']
        self.used_gam = self.train_args['used_GAM']
        self.used_mmfm = self.train_args['used_MMFM']


        self.rank = 0


        torch.backends.cudnn.benchmark = True

        start_time = time.time()


        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        if self.data_args['type'] == 32:
            self.tensor_type = torch.float32
        elif self.data_args['type'] == 64:
            self.tensor_type = torch.float64
        elif self.data_args['type'] == 16:
            self.tensor_type = torch.float16
        else:
            self.tensor_type = torch.float32


        self.model = Affordsplat_net(self.model_args).to(self.device)
        print("Model initialized.")


        if self.ckpt_path:
            if not os.path.exists(self.ckpt_path):
                raise FileNotFoundError(f"Checkpoint {self.ckpt_path} is not found")
            checkpoint = torch.load(self.ckpt_path)
            if "mmfm.pos_embed.weight" in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"]["mmfm.pos_embed.weight"]
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model's pretrain weights load successfully. Checkpoints path:{self.ckpt_path}")
        elif self.ckpt_path == None:
            raise FileNotFoundError(f"Checkpoint {self.ckpt_path} is not found")

        print("Distributed training model initialized successfully")



    def read_gs(self,path):
        gs_data=plyfile.PlyData.read(path)['vertex']


        gs_x=list(gs_data['x'])
        gs_y=list(gs_data['y'])
        gs_z=list(gs_data['z'])
        gs_mean=np.column_stack((gs_x,gs_y,gs_z))
        gs_mean_norm,_,_=mean_normalize(gs_mean)


        gs_scale_0=list(gs_data['scale_0'])
        gs_scale_1=list(gs_data['scale_1'])
        gs_scale_2=list(gs_data['scale_2'])
        gs_scale=np.column_stack((gs_scale_0,gs_scale_1,gs_scale_2))
        gs_rot_0=list(gs_data['rot_0'])
        gs_rot_1=list(gs_data['rot_1'])
        gs_rot_2=list(gs_data['rot_2'])
        gs_rot_3=list(gs_data['rot_3'])
        gs_rot=np.column_stack((gs_rot_0,gs_rot_1,gs_rot_2,gs_rot_3))


        gs_features=np.column_stack((gs_mean_norm,gs_scale,gs_rot))

        return gs_features, gs_mean


    def data_process(self,question, answer, origin_ply_path):
        gs_features, _ = self.read_gs(origin_ply_path)  # [N,10]
        gs_features = np.expand_dims(gs_features, axis=0)  # [1,N,10]

        question_list = [question]
        answer_list = [answer]

        return gs_features, question_list, answer_list


    def get_result(self, obj, aff, question, answer, origin_ply_path):
        with torch.no_grad():
            self.model.eval()

            gs_features, question, answer = self.data_process(question, answer, origin_ply_path)


            gs_features=torch.tensor(np.array(gs_features),dtype=self.tensor_type).to(self.device) # [1,N,10]
            masks = torch.ones((1,gs_features.shape[1]),dtype=self.tensor_type).to(self.device) # [1,N]
            pc_mean_all = torch.randn((1,1,2048, 3),dtype=self.tensor_type).to(self.device)
            pc_aff_map_all = torch.ones((1,1,2048, 3),dtype=self.tensor_type).to(self.device)


            (dynamic_kernels,           # [1, gs_embed_dim, 1]
             pred_aff_map,              # [1, N, 1]
             text_loss,                 # tensor
             predicted_text,            # list,[1]
             ) = self.model(
                gs_features, gs_features, masks, pc_mean_all, pc_aff_map_all, question, answer, device=self.device,use_csa=False)


            pred_aff_map = pred_aff_map.flatten().cpu().detach().numpy() # [N]
            predicted_text = predicted_text[0] # str


            self.write_pred_aff_ply(obj,aff,origin_ply_path, pred_aff_map)

            print(f"{obj}-{aff}:")
            print(f"Question:{question}")
            print(f"Answer:{predicted_text}")



    def write_pred_aff_ply(self,obj, aff, origin_ply_path, pred_aff_map):
        results_path = f'/root/autodl-tmp/AffordSplat/results/{obj}/'
        if not os.path.exists(results_path):
            os.makedirs(results_path,exist_ok=True)

        file_id = origin_ply_path.split('/')[-1].split('_')[-1].split('.')[0]

        filtered_mask = pred_aff_map >= 0.5 # 二值掩码
        indices = np.where(filtered_mask)[0] # 大于0.5的位置
        p_values = pred_aff_map[indices]

        # 2. 构建颜色过渡曲线
        # 红色通道：0.5时255，1时保持255
        # 绿色/蓝色通道：0.5时100 → 1时0
        alpha = (p_values - 0.5) * 2  # 将0.5-1映射到0-1

        plydata = PlyData.read(origin_ply_path)

        vertex_data = plydata['vertex'].data.copy()  # 创建副本避免修改原始数据

        # 更新颜色分量（保持原色通道数据类型）
        vertex_data['f_dc_0'][indices] = 255 # 红色通道保持最大值
        # vertex_data['f_dc_0'][indices] = 255
        # vertex_data['f_dc_1'][indices] = np.clip(50 * (1-alpha), 0, 255)
        # vertex_data['f_dc_2'][indices] = np.clip(100 * (1-alpha), 0, 255)


        # 可选：保留原始位置数据（若需要过滤低概率点）
        if self.apply_threshold:  # 添加类属性控制是否过滤
            vertex_data = vertex_data[filtered_mask]

        # 构建新PLY结构
        new_vertex = PlyElement.describe(
            vertex_data,
            'vertex',
            val_types=[(prop.name, prop.val_dtype) for prop in plydata['vertex'].properties]
        )

        # 写入文件（保留二进制格式）
        PlyData([new_vertex], text=False).write(os.path.join(results_path+f"{obj}_{aff}_{file_id}_result.ply"))
        print(f"{obj}-{aff}'s visualization result has been saved to {os.path.join(results_path, f"{obj}_{aff}_result.ply")}")


if __name__ == '__main__':

    apply_threshold = True #是否将掩码直接用于原文件，还是输出掩码位置
    # 问题
    obj = 'storagefurniture'
    aff = 'contain'
    question = "Identify the key points on the storagefurniture that ensure a successful containing experience."
    answer = "<Aff>"
    original_path = "/root/autodl-tmp/AffordSplat/AffordSplat/Seen/train/storagefurniture/Gaussian/GS_0030.ply"

    get_ply_result = Get_ply_result(apply_threshold)
    get_ply_result.get_result(obj, aff, question, answer, original_path)
