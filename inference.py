import os
import numpy as np
from plyfile import PlyData, PlyElement
import time
import yaml
import copy
from tqdm import tqdm
import shutil
from sklearn.metrics import *
from peft import LoraConfig, get_peft_model,PeftModel, PeftConfig
from collections import defaultdict

from torch import nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from data.dataloader import *
from data.collate_fn import *
from utils.logger import logger_init
from utils.loss_function import *
from models.Affordsplat_net import *
from models.GPSE import gpse
from utils.evaluate_function import *

# 读取配置文件
def config_read(path):
    with open(path, 'r', encoding='utf-8') as f:
        args = yaml.safe_load(f)
    return args


# inference
class inferencer():
    def __init__(self):

        # read configuration
        self.model_args = config_read('config/model_config.yaml')
        self.data_args = config_read('config/data_config.yaml')
        self.train_args = config_read('config/train_config.yaml')

        self.inference_args = self.train_args['inference']

        self.ckpt_path = self.inference_args['ckpt_path'] # Weight Path
        self.gpu_num = self.train_args['gpu_num'] # Number of GPUs used
        self.setting = self.data_args['setting'] # Use Seen or Unseen configuration
        self.used_gam = self.train_args['used_GAM']
        self.used_mmfm = self.train_args['used_MMFM']


        # Get the local rank of the current process. In the case of a single machine, rank and local_rank are the same
        self.local_rank = int(os.environ["LOCAL_RANK"])

        # DDP backend初始化
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=self.gpu_num,
                                rank=int(os.environ["RANK"]))

        self.rank = dist.get_rank()

        torch.backends.cudnn.benchmark = True

        os.environ['OMP_NUM_THREADS'] = str(self.data_args['num_workers'])

        start_time = time.time()

        self.device = self.local_rank

        self.logger = logger_init(self.rank,"inference",self.setting, is_resume = False)

        if self.data_args['type'] == 32:
            self.tensor_type = torch.float32
        elif self.data_args['type'] == 64:
            self.tensor_type = torch.float64
        elif self.data_args['type'] == 16:
            self.tensor_type = torch.float16
        else:
            self.tensor_type = torch.float32

        start_time = self.init_dataset(start_time)


        self.model = Affordsplat_net(self.model_args).to(self.local_rank)
        self.logger.info("Model initialized.")


        if dist.get_rank() == 0 and self.ckpt_path:
            if not os.path.exists(self.ckpt_path):
                raise FileNotFoundError(f"Checkpoint {self.ckpt_path} is not found")
            checkpoint = torch.load(self.ckpt_path)
            if "mmfm.pos_embed.weight" in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"]["mmfm.pos_embed.weight"]
            self.model.load_state_dict(checkpoint['model_state_dict'])
            start_time = self.log_with_time(start_time,f"Model's pretrain weights load successfully. Checkpoints path:{self.ckpt_path}")
        elif self.ckpt_path == None:
            raise FileNotFoundError(f"Checkpoint {self.ckpt_path} is not found")


        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],find_unused_parameters=True)

        start_time = self.log_with_time(start_time, "Distributed training model initialized successfully")


        self.obj_id_to_name = {
            0: "trashcan",1: "clock",2: "keyboard",3: "bed",4: "dishwasher",
            5: "knife",6: "bottle",7: "chair",8: "bag",9: "laptop",
            10: "table",11: "faucet",12: "earphone",13: "storagefurniture",14: "hat",
            15: "microwave",16: "mug",17: "bowl",18: "vase",19: "door",
            20: "display"
        }
        self.aff_id_to_name = {
            0: "pour",1: "contain",2: "open",3: "display",4: "press",
            5: "lay",6: "support",7: "sit",8: "stab",9: "grasp",
            10: "cut",11: "wrap_grasp",12: "move",13: "lift",14: "listen",
            15: "wear",16: "pull",17: "push"
        }
        self.obj_name_to_id = {v: k for k, v in self.obj_id_to_name.items()}
        self.aff_name_to_id = {v: k for k, v in self.aff_id_to_name.items()}

        if self.setting == 'Seen':
            self.obj_to_aff = json.load(open("/root/autodl-tmp/AffordSplat/AffordSplat/obj_aff_structure.json"))
        elif self.setting == 'Unseen':
            self.obj_to_aff = json.load(open("/root/autodl-tmp/AffordSplat/AffordSplat/UnSeen_test.json"))
        else:
            raise ValueError("setting must be Seen or Unseen")


    def init_dataset(self,start_time):


        self.dataset_test = QAffordSplat_val_test(setting=self.data_args['setting'],
                                                  root_dir=self.data_args['root_dir'],
                                                  state='test')

        test_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset_test)

        self.dataloader_test = data.DataLoader(self.dataset_test,
                                              batch_size=self.data_args['batch_size'],
                                              shuffle=self.data_args['shuffle'],
                                              drop_last=self.data_args['drop_last'],
                                              collate_fn = collate_fn_val_test,
                                              num_workers=self.data_args['num_workers'],
                                              sampler = test_sampler)

        start_time = self.log_with_time(start_time, "Test dataloader builded up,len(AffordSplat)={}".format(self.dataset_test.__len__()))

        return start_time



    def log_with_time(self,start_time, message):
        if self.rank == 0:
            elapsed_time = time.time() - start_time
            self.logger.info(f"{message} | Time consumed: {elapsed_time:.2f}s")
        return time.time()


    def eval(self):
        self.logger.info(f'————————————————EVALUATION START————————————————')
        self.logger.info(f'  - Test Dataloader Length: {len(self.dataloader_test)}')
        self.logger.info(f'  - Test Dataloader Setting: {self.setting}')
        self.logger.info(f"  - World Size: {dist.get_world_size()}")
        self.logger.info(f'  - Evaluation Metrics: MAE, SIM, KLD, AUC, IOU')


        total_metrics = {
            'mae': 0.0,
            'sim': 0.0,
            'kld': 0.0,
            'auc': 0.0,
            'iou': 0.0
        }


        metrics_by_comb = defaultdict(lambda: {
            'mae_sum': 0.0, 'mae_points': 0,
            'sim_sum': 0.0, 'kld_sum': 0.0,
            'auc_sum': 0.0, 'iou_sum': 0.0,
            'data_count': 0
        })

        total_points = 0
        total_data = 0
        eval_time = time.time()
        batch_time = time.time()

        with torch.no_grad():
            self.model.eval()

            for i, (obj,                # [Batch size]
                    aff,                # [Batch size]
                    gs_features,        # [Batsh size, batch_min_points, 10]
                    padded_gs_datas,    # [Batsh size, batch_max_points, 10]
                    masks,              # [Batsh size, batch_max_points]
                    gs_aff_maps,        # [Batch size, batch_max_points, 1]
                    question,           # [Batch size,15]
                    answer,             # [Batch size]
                    original_gs_path    # [Batch size]
                    ) in enumerate(self.dataloader_test):

                # 数据放在gpu上
                gs_features=torch.tensor(np.array(gs_features),dtype=self.tensor_type).to(self.device)
                padded_gs_datas=torch.tensor(np.array(padded_gs_datas),dtype=self.tensor_type).to(self.device)
                masks = torch.tensor(np.array(masks),dtype=self.tensor_type).to(self.device)
                gs_aff_maps=torch.tensor(np.array(gs_aff_maps),dtype=self.tensor_type).to(self.device)
                pc_mean_all = torch.randn((self.data_args['batch_size'],1,2048, 3),dtype=self.tensor_type).to(self.device)
                pc_aff_map_all = torch.ones((self.data_args['batch_size'],1,2048, 3),dtype=self.tensor_type).to(self.device)



                (dynamic_kernels,           # [Batch_size, gs_embed_dim, 1]
                 pred_aff_map,              # [Batch_size, batch_max_points, 1]
                 text_loss,                 # tensor
                 predicted_text,            # list,[Batch_size]
                 ) = self.model(
                    gs_features, padded_gs_datas, masks, pc_mean_all, pc_aff_map_all, question, answer, device=self.device,use_csa=False)


                batch_error,batch_points = MAE(pred_aff_map,gs_aff_maps,masks)  # [Batch size]，[Batch size]
                batch_sim = SIM(pred_aff_map,gs_aff_maps,masks)                 # [Batch_size]
                batch_kld = KLD(pred_aff_map,gs_aff_maps,masks)                 # [Batch_size]
                batch_auc = AUC(pred_aff_map, gs_aff_maps, masks)               # [Batch_size]
                batch_iou = IOU(pred_aff_map, gs_aff_maps, masks)               # [Batch_size]

                for j in range(len(obj)):
                    current_obj = obj[j]
                    current_aff = aff[j]
                    key = (current_obj, current_aff)


                    sample_mae = batch_error[j].item()
                    sample_points = batch_points[j].item()
                    sample_sim = batch_sim[j].item()
                    sample_kld = batch_kld[j].item()
                    sample_auc = batch_auc[j].item()
                    sample_iou = batch_iou[j].item()


                    metrics_by_comb[key]['mae_sum'] += sample_mae
                    metrics_by_comb[key]['mae_points'] += sample_points
                    metrics_by_comb[key]['sim_sum'] += sample_sim
                    metrics_by_comb[key]['kld_sum'] += sample_kld
                    metrics_by_comb[key]['auc_sum'] += sample_auc
                    metrics_by_comb[key]['iou_sum'] += sample_iou
                    metrics_by_comb[key]['data_count'] += 1



                total_metrics['sim'] += torch.sum(batch_sim).item()
                total_metrics['kld'] += torch.sum(batch_kld).item()
                total_metrics['auc'] += torch.sum(batch_auc).item()
                total_metrics['iou'] += torch.sum(batch_iou).item()
                total_metrics['mae'] += torch.sum(batch_error).item()
                total_points += torch.sum(batch_points).item()
                total_data+=len(obj)


                batch_mae = torch.sum(batch_error)/torch.sum(batch_points)

                if (i+1) % 5 == 0:
                    batch_time = self.log_with_time(batch_time,
                                                    f'Batch {i + 1}/{len(self.dataloader_test)} | '
                                                    f'MAE: {batch_mae.item():.4f} | '
                                                    f'SIM: {torch.mean(batch_sim).item():.4f} | '
                                                    f'KLD: {torch.mean(batch_kld).item():.4f} | '
                                                    f'AUC: {torch.mean(batch_auc).item():.4f} | '
                                                    f'IOU: {torch.mean(batch_iou).item():.4f}')


            dist.barrier()
            metrics_tensor = torch.tensor([
                total_metrics['mae'],
                total_metrics['sim'],
                total_metrics['kld'],
                total_metrics['auc'],
                total_metrics['iou'],
                total_points,
                total_data
            ], dtype=self.tensor_type, device=self.device)


            obj_aff_combin = [
                (self.obj_name_to_id[obj], self.aff_name_to_id[aff])
                for obj in self.obj_to_aff.keys()
                for aff in self.obj_to_aff[obj]
            ]

            local_metrics = {
                (obj_id, aff_id): [
                    obj_id,
                    aff_id,
                    0, 0, 0, 0, 0, 0, 0
                ]
                for obj_id, aff_id in obj_aff_combin}


            for key, values in list(metrics_by_comb.items()):
                obj, aff = key
                obj_id = self.obj_name_to_id[obj]
                aff_id = self.aff_name_to_id[aff]
                id_key = (obj_id, aff_id)
                local_metrics[id_key]= [
                    obj_id,
                    aff_id,
                    values['mae_sum'], values['mae_points'],
                    values['sim_sum'], values['kld_sum'],
                    values['auc_sum'], values['iou_sum'],
                    values['data_count']
                    ]

            local_tensor_data = [local_metrics[(self.obj_name_to_id[obj],self.aff_name_to_id[aff])] for obj in self.obj_to_aff for aff in self.obj_to_aff[obj]]
            local_tensor = torch.tensor(local_tensor_data, dtype=self.tensor_type, device=self.device)


            all_metrics = [torch.zeros_like(metrics_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(all_metrics, metrics_tensor)

            dist.barrier()

            all_tensors = [torch.zeros_like(local_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(all_tensors, local_tensor)

            dist.barrier()



            if dist.get_rank() == 0:
                total_mae = 0.0
                total_sim = 0.0
                total_kld = 0.0
                total_auc = 0.0
                total_iou = 0.0
                total_points = 0
                total_data = 0


                for metrics in all_metrics:
                    total_mae += metrics[0].item()
                    total_sim += metrics[1].item()
                    total_kld += metrics[2].item()
                    total_auc += metrics[3].item()
                    total_iou += metrics[4].item()
                    total_points += metrics[5].item()
                    total_data += metrics[6].item()


                total_metrics['mae'] = total_mae / total_points
                total_metrics['sim'] = total_sim / total_data
                total_metrics['kld'] = total_kld / total_data
                total_metrics['auc'] = total_auc / total_data
                total_metrics['iou'] = total_iou / total_data


                self.log_with_time(eval_time,f'Overall Evaluation Metrics')
                self.logger.info(f'  - MAE: {total_metrics["mae"]:.4f}')
                self.logger.info(f'  - SIM: {total_metrics["sim"]:.4f}')
                self.logger.info(f'  - KLD: {total_metrics["kld"]:.4f}')
                self.logger.info(f'  - AUC: {total_metrics["auc"]:.4f}')
                self.logger.info(f'  - IOU: {total_metrics["iou"]:.4f}')


                global_metrics = defaultdict(lambda: {
                                    'mae_sum': 0.0,
                                    'mae_points': 0,
                                    'sim_sum': 0.0,
                                    'kld_sum': 0.0,
                                    'auc_sum': 0.0,
                                    'iou_sum': 0.0,
                                    'data_count': 0
                                                            })


                obj_metrics = defaultdict(lambda: {
                    'mae_sum': 0.0, 'mae_points': 0, 'sim_sum': 0.0,
                    'kld_sum': 0.0, 'auc_sum': 0.0, 'iou_sum': 0.0,
                    'data_count': 0
                })


                aff_metrics = defaultdict(lambda: {
                    'mae_sum': 0.0, 'mae_points': 0, 'sim_sum': 0.0,
                    'kld_sum': 0.0, 'auc_sum': 0.0, 'iou_sum': 0.0,
                    'data_count': 0
                })

                for tensor in all_tensors:
                    for row in tensor:
                        obj_id, aff_id = int(row[0]), int(row[1])
                        key = (obj_id, aff_id)
                        global_metrics[key]['mae_sum'] += row[2]
                        global_metrics[key]['mae_points'] += row[3]
                        global_metrics[key]['sim_sum'] += row[4]
                        global_metrics[key]['kld_sum'] += row[5]
                        global_metrics[key]['auc_sum'] += row[6]
                        global_metrics[key]['iou_sum'] += row[7]
                        global_metrics[key]['data_count'] += row[8]


                        obj_metrics[obj_id]['mae_sum'] += row[2]
                        obj_metrics[obj_id]['mae_points'] += row[3]
                        obj_metrics[obj_id]['sim_sum'] += row[4]
                        obj_metrics[obj_id]['kld_sum'] += row[5]
                        obj_metrics[obj_id]['auc_sum'] += row[6]
                        obj_metrics[obj_id]['iou_sum'] += row[7]
                        obj_metrics[obj_id]['data_count'] += row[8]


                        aff_metrics[aff_id]['mae_sum'] += row[2]
                        aff_metrics[aff_id]['mae_points'] += row[3]
                        aff_metrics[aff_id]['sim_sum'] += row[4]
                        aff_metrics[aff_id]['kld_sum'] += row[5]
                        aff_metrics[aff_id]['auc_sum'] += row[6]
                        aff_metrics[aff_id]['iou_sum'] += row[7]
                        aff_metrics[aff_id]['data_count'] += row[8]


                for (obj_id, aff_id), metrics in global_metrics.items():
                    mae_avg = metrics['mae_sum'] / metrics['mae_points'] if metrics['mae_points'] > 0 else 0
                    sim_avg = metrics['sim_sum'] / metrics['data_count'] if metrics['data_count'] > 0 else 0
                    kld_avg = metrics['kld_sum'] / metrics['data_count']
                    auc_avg = metrics['auc_sum'] / metrics['data_count']
                    iou_avg = metrics['iou_sum'] / metrics['data_count']


                    obj = self.obj_id_to_name[obj_id]
                    aff = self.aff_id_to_name[aff_id]
                    self.logger.info(
                        f"组合 ({obj}, {aff}): "
                        f"MAE={mae_avg:.4f} | SIM={sim_avg:.4f} | "
                        f"KLD={kld_avg:.4f} | AUC={auc_avg:.4f} | "
                        f"IOU={iou_avg:.4f} | 样本数={metrics['data_count']}"
                    )


                for obj_id, metrics in obj_metrics.items():
                    mae_avg = metrics['mae_sum'] / metrics['mae_points'] if metrics['mae_points'] > 0 else 0
                    sim_avg = metrics['sim_sum'] / metrics['data_count'] if metrics['data_count'] > 0 else 0
                    kld_avg = metrics['kld_sum'] / metrics['data_count']
                    auc_avg = metrics['auc_sum'] / metrics['data_count']
                    iou_avg = metrics['iou_sum'] / metrics['data_count']

                    obj_name = self.obj_id_to_name[obj_id]
                    self.logger.info(
                        f"对象 {obj_name}: "
                        f"MAE={mae_avg:.4f} | SIM={sim_avg:.4f} | "
                        f"KLD={kld_avg:.4f} | AUC={auc_avg:.4f} | "
                        f"IOU={iou_avg:.4f} | 样本数={metrics['data_count']}"
                    )


                for aff_id, metrics in aff_metrics.items():
                    mae_avg = metrics['mae_sum'] / metrics['mae_points'] if metrics['mae_points'] > 0 else 0
                    sim_avg = metrics['sim_sum'] / metrics['data_count'] if metrics['data_count'] > 0 else 0
                    kld_avg = metrics['kld_sum'] / metrics['data_count']
                    auc_avg = metrics['auc_sum'] / metrics['data_count']
                    iou_avg = metrics['iou_sum'] / metrics['data_count']

                    aff_name = self.aff_id_to_name[aff_id]
                    self.logger.info(
                        f"动作 {aff_name}: "
                        f"MAE={mae_avg:.4f} | SIM={sim_avg:.4f} | "
                        f"KLD={kld_avg:.4f} | AUC={auc_avg:.4f} | "
                        f"IOU={iou_avg:.4f} | 样本数={metrics['data_count']}"
                    )


        dist.barrier()
        self.log_with_time(eval_time,f'————————————————EVALUATION END————————————————')


    def inference(self):
        self.eval()


if __name__ == '__main__':
    try:
        inferencer = inferencer()
        inferencer.inference()
    except Exception as e:

        dist.destroy_process_group()
        raise


    dist.destroy_process_group()