import time
import yaml
import copy
from tqdm import tqdm
import shutil
from sklearn.metrics import *
from peft import LoraConfig, get_peft_model,PeftModel, PeftConfig

from torch import nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from data.dataloader import *
from data.collate_fn import *
from utils.logger import logger_init
from utils.loss_function import *
from models.Affordsplat_net import Affordsplat_net
from models.GPSE import gpse
from utils.evaluate_function import *
from utils.loss_function import *


def config_read(path):
    with open(path, 'r', encoding='utf-8') as f:
        args = yaml.safe_load(f)
    return args


class pretrain_trainer():
    def __init__(self):


        self.model_args = config_read('config/model_config.yaml')
        self.data_args = config_read('config/data_config.yaml')
        self.train_args = config_read('config/train_config.yaml')

        self.train_pretrain_args = self.train_args['pretrain']
        self.train_lora_args = self.train_args['lora']
        self.data_pretrain_args = self.data_args['pretrain']

        self.ckpt_epoch = self.train_pretrain_args['ckpt_epoch']
        self.ckpt_path = self.train_pretrain_args['ckpt_path']
        self.max_checkpoints = self.train_pretrain_args['max_checkpoints']
        self.is_resume = self.train_pretrain_args['is_resume']
        self.gpu_num = self.train_args['gpu_num']
        self.setting = self.data_args['setting']

        writer_path = 'runs/affordsplat/pretrain/' + self.setting
        if os.path.exists(writer_path) and not self.is_resume:
            shutil.rmtree(writer_path)
        os.makedirs(writer_path,exist_ok=True)
        self.writer = SummaryWriter(writer_path)


        self.local_rank = int(os.environ["LOCAL_RANK"])


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


        self.logger = logger_init(self.rank,"pretrain",self.setting,self.is_resume)


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


        if dist.get_rank() == 0 and self.ckpt_path:
            if not os.path.exists(self.ckpt_path):
                raise FileNotFoundError(f"Checkpoint {self.ckpt_path} is not found")
            checkpoint = torch.load(self.ckpt_path)
            if "mmfm.pos_embed.weight" in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"]["mmfm.pos_embed.weight"]
            self.model.load_state_dict(checkpoint['model_state_dict'])
            start_time = self.log_with_time(start_time,f"Model's pretrain weights load successfully. Checkpoints path:{self.ckpt_path}")


        self.lora_config = LoraConfig(
            r=self.train_lora_args['rank'],
            lora_alpha=self.train_lora_args['alpha'],
            lora_dropout=self.train_lora_args['dropout'],
            bias=self.train_lora_args['bias'],
            target_modules=self.train_lora_args['target_modules'],
            task_type=self.train_lora_args['task_type'],
            inference_mode=False
        )

        llm = self.model.mllm.model
        llm = get_peft_model(llm, self.lora_config)

        self.model.mllm.model = llm
        self.logger.info("Successfully configured language model for Lora fine-tuning")
        if self.rank == 0:
            llm.print_trainable_parameters()



        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],find_unused_parameters=True)

        start_time = self.log_with_time(start_time, "Distributed training model initialized successfully,")


        self.best_iou = 0


    def init_dataset(self,start_time):

        self.dataset_pretrain = QAffordSplat_train(pc_num=self.data_pretrain_args['pc_num'],
                                           setting=self.data_args['setting'],
                                           is_pretrain=self.data_pretrain_args['is_pretrain'],
                                           gs_label_num=self.data_pretrain_args['gs_label_num'],
                                           random_seed=self.data_pretrain_args['random_seed'],
                                           root_dir=self.data_args['root_dir'])

        pretrain_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset_pretrain)

        self.dataloader_pretrain = data.DataLoader(self.dataset_pretrain,
                                           batch_size=self.data_args['batch_size'],
                                           shuffle=self.data_args['shuffle'],
                                           drop_last=self.data_args['drop_last'],
                                           sampler=pretrain_sampler,
                                           collate_fn=collate_fn_train,
                                           num_workers=self.data_args['num_workers'])


        start_time = self.log_with_time(start_time,"Pretrain dataloader builded up,len(AffordSplat)={}".format(self.dataset_pretrain.__len__()))


        self.dataset_val = QAffordSplat_val_test(setting=self.data_args['setting'],
                                                 root_dir=self.data_args['root_dir'])

        val_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset_val)

        self.dataloader_val = data.DataLoader(self.dataset_val,
                                              batch_size=self.data_args['batch_size'],
                                              shuffle=self.data_args['shuffle'],
                                              drop_last=self.data_args['drop_last'],
                                              collate_fn=collate_fn_val_test,
                                              num_workers=self.data_args['num_workers'],
                                              sampler=val_sampler)

        start_time = self.log_with_time(start_time, "Val dataloader builded up,len(AffordSplat)={}".format(self.dataset_val.__len__()))

        return start_time



    def log_with_time(self,start_time, message):
        if self.rank == 0:
            elapsed_time = time.time() - start_time
            self.logger.info(f"{message} | Time consumed: {elapsed_time:.2f}s")
        return time.time()



    def save_checkpoint(self,epoch, overall_mean_loss):
        if self.rank==0:
            if (epoch + 1) % self.ckpt_epoch == 0:
                merged_model = copy.deepcopy(self.model.module)
                if isinstance(merged_model.mllm.model, PeftModel):
                    merged_model.mllm.model = merged_model.mllm.model.merge_and_unload()


                save_dir = './checkpoints/pretrain/'+self.setting
                os.makedirs(save_dir, exist_ok=True)

                checkpoint_path = os.path.join(save_dir, f"pretrain_"+self.setting+f"_{epoch}.pth")
                if os.path.exists(checkpoint_path):
                    self.logger.warning(f"Checkpoint {checkpoint_path} already exists, overwriting!")

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': merged_model.state_dict(),
                    'optimizer_state_dict': self.pretrain_optimizer.state_dict(),
                    'scheduler_state_dict': self.pretrain_scheduler.state_dict(),
                    'overall_mean_loss': overall_mean_loss
                }, checkpoint_path)
                self.logger.info(f"Epoch:{epoch} | Checkpoint saved to {checkpoint_path}")


                checkpoints = [f for f in os.listdir(save_dir) if f.startswith("pretrain_") and f.endswith(".pth")]
                normal_checkpoints = [f for f in checkpoints if not f.endswith('best.pth')]

                normal_checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

                if len(normal_checkpoints) > self.max_checkpoints:
                    oldest_checkpoint = normal_checkpoints[0]
                    os.remove(os.path.join(save_dir, oldest_checkpoint))
                    self.logger.info(f"Deleted oldest checkpoint: {oldest_checkpoint}")


    def pretrain(self):

        self.pretrain_optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.train_pretrain_args['lr'],
                                          betas=(0.9, 0.999),
                                          eps=1e-8,
                                          weight_decay=self.train_pretrain_args['weight_decay'])


        self.total_epoch = self.train_pretrain_args['epochs']
        self.start_epoch = 0


        scheduler_state = None
        if self.is_resume:

            optimizer_state_list = [None]
            start_epoch_list = [0]
            scheduler_state_list = [None]

            if self.rank == 0 and self.ckpt_path:
                if not os.path.exists(self.ckpt_path):
                    raise FileNotFoundError(f"Checkpoint {self.ckpt_path} not found")


                checkpoint = torch.load(self.ckpt_path, map_location='cpu')


                required_keys = ['optimizer_state_dict', 'epoch', 'scheduler_state_dict']
                if not all(k in checkpoint for k in required_keys):
                    raise ValueError("Checkpoint missing required keys")

                optimizer_state_list[0] = checkpoint['optimizer_state_dict']
                start_epoch_list[0] = checkpoint['epoch']
                scheduler_state_list[0] = checkpoint.get('scheduler_state_dict', None)

            dist.barrier()


            dist.broadcast_object_list(optimizer_state_list, src=0)
            dist.broadcast_object_list(start_epoch_list, src=0)
            dist.broadcast_object_list(scheduler_state_list, src=0)

            optimizer_state = optimizer_state_list[0]
            start_epoch_state = start_epoch_list[0]
            scheduler_state = scheduler_state_list[0]


            if optimizer_state is not None:
                self.pretrain_optimizer.load_state_dict(optimizer_state)
            else:
                self.logger.warning("Optimizer state is None, skip loading.")
            self.start_epoch = start_epoch_state if start_epoch_state is not None else 0


        self.pretrain_scheduler = torch.optim.lr_scheduler.StepLR(self.pretrain_optimizer,
                                                                  step_size=self.train_pretrain_args['step_size'],
                                                                  gamma=self.train_pretrain_args['gamma'])

        if scheduler_state is not None:
            self.pretrain_scheduler.load_state_dict(scheduler_state)

        if self.is_resume:
            self.logger.info(f"Successfully resume training from {self.ckpt_path}")


        self.logger.info(f'————————————————PRETRAIN START————————————————')

        self.logger.info(f"Training starts with the following configurations:")
        self.logger.info(f"  - Total Epochs: {self.total_epoch}")
        self.logger.info(f"  - Start Epoch: {self.start_epoch}")
        self.logger.info(f"  - Learning Rate: {self.train_pretrain_args['lr']}")
        self.logger.info(f"  - Weight Decay: {self.train_pretrain_args['weight_decay']}")
        self.logger.info(f"  - Optimizer: Adam")
        self.logger.info(f"  - Scheduler: StepLR(step_size={self.train_pretrain_args['step_size']}, gamma={self.train_pretrain_args['gamma']})")
        self.logger.info(f"  - World Size: {dist.get_world_size()}")
        self.logger.info(f"  - Every {self.ckpt_epoch} epochs save a checkpoints file")
        self.logger.info(f"  - Training Dataloader Length: {len(self.dataloader_pretrain)}")
        self.logger.info(f"  - Training Dataloader Setting: {self.setting}")

        pretrain_time = time.time()


        iterator = tqdm(
            range(self.start_epoch,self.total_epoch),
            desc="Training",
            disable=self.rank != 0
        )

        for epoch in iterator:


            self.dataloader_pretrain.sampler.set_epoch(epoch)

            loss_sum = 0.0
            data_num = 0
            epoch_time = time.time()
            self.model.train()

            for i, (obj,                # [Batch size]
                    aff,                # [Batch size]
                    gs_features,        # [Batch size,batch_min_points,gs_C(10)]
                    padded_gs_datas,    # [Batch size,batch_max_points,gs_C(10)]
                    mask_batch,         # [Batch size,batch_max_points]
                    gs_aff_maps,        # [Batch size](None)
                    pc_mean_all,        # [Batch size,pc_num,points_num,pc_C(3)]
                    pc_aff_map_all,     # [Batch size,pc_num,points_num,1]
                    question,           # [Batch size]
                    answer              # [Batch size]
                    ) in enumerate(self.dataloader_pretrain):

                self.pretrain_optimizer.zero_grad()

                batch_time = time.time()


                padded_gs_means = np.array(padded_gs_datas)[:, :, :3]
                loss_consis_weight = gpse(padded_gs_means,
                                          np.array(pc_mean_all),
                                          np.array(mask_batch),
                                          T=np.float32(self.train_args['gspe']['T']),
                                          type=self.train_args['gspe']['type'])         # [batch_size, pc_num]


                gs_features=torch.tensor(np.array(gs_features),dtype=self.tensor_type).to(self.device)
                padded_gs_datas=torch.tensor(np.array(padded_gs_datas),dtype=self.tensor_type).to(self.device)
                mask_batch = torch.tensor(np.array(mask_batch),dtype=self.tensor_type).to(self.device)
                pc_mean_all=torch.tensor(np.array(pc_mean_all),dtype=self.tensor_type).to(self.device)
                pc_aff_map_all=torch.tensor(np.array(pc_aff_map_all),dtype=self.tensor_type).to(self.device)
                loss_consis_weight=torch.tensor(loss_consis_weight,dtype=self.tensor_type).to(self.device)



                (dynamic_kernels,           # [Batch_size, gs_embed_dim, 1]
                 pred_aff_map,              # [Batch_size, batch_max_points, 1]
                 text_loss,                 # tensor
                 predicted_text,            # list,[Batch_size]
                 gs_stru_features,          # [Batch_size, nsample, embed_dim]
                 pc_stru_features           # [Batch_size, pc_num, points_num, embed_dim]
                 ) = self.model(
                    gs_features, padded_gs_datas, mask_batch, pc_mean_all, pc_aff_map_all, question, answer, device=self.device)


                loss_consis = cosine_similarity_loss(gs_stru_features, pc_stru_features) # [Batch size, pc_num, 1]
                loss_consis = torch.sum(loss_consis_weight.unsqueeze(-1) * loss_consis) # tensor
                loss = loss_consis + text_loss


                loss.backward()
                self.pretrain_optimizer.step()

                loss_sum +=loss.item()
                data_num += len(obj)

                if self.rank == 0:
                    print(f'Epoch:{epoch}| iteration:{i}|{len(self.dataloader_pretrain)}  | time consumed:{time.time() - batch_time}')
                    print(f'loss:{loss.item() / len(obj)} | loss_consis:{loss_consis.item() / len(obj)} | text_loss:{text_loss.item() / len(obj)}')



            overall_mean_loss = loss_sum / data_num


            if self.rank == 0:

                self.writer.add_scalar('overall_mean_loss', overall_mean_loss, epoch)
                self.save_checkpoint(epoch,overall_mean_loss)

            dist.barrier()

            self.eval(epoch)

            self.pretrain_scheduler.step()

            self.logger.info(
                f'Epoch:{epoch} | Overall mean_loss:{overall_mean_loss:.4f} | Time consumed:{time.time() - epoch_time:.2f}s')
        self.log_with_time(pretrain_time,f'————————————————PRETRAIN END————————————————')


    def eval(self,epoch):
        self.logger.info(f'————————————————EVALUATION START————————————————')
        self.logger.info(f'  - Val Dataloader Length: {len(self.dataloader_val)}')
        self.logger.info(f'  - Val Dataloader Setting: {self.setting}')
        self.logger.info(f"  - World Size: {dist.get_world_size()}")
        self.logger.info(f'  - Evaluation Metrics: MAE, SIM, KLD, AUC, IOU')

        # 初始化统计数据结构
        total_metrics = {
            'loss': 0.0,
            'mae': 0.0,
            'sim': 0.0,
            'kld': 0.0,
            'auc': 0.0,
            'iou': 0.0
        }
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
                    answer              # [Batch size]
                    ) in enumerate(self.dataloader_val):



                gs_features=torch.tensor(np.array(gs_features),dtype=self.tensor_type).to(self.device)
                padded_gs_datas=torch.tensor(np.array(padded_gs_datas),dtype=self.tensor_type).to(self.device)
                masks = torch.tensor(np.array(masks),dtype=self.tensor_type).to(self.device)
                gs_aff_maps=torch.tensor(np.array(gs_aff_maps),dtype=self.tensor_type).to(self.device)
                pc_mean_all = torch.randn((self.data_args['batch_size'],self.data_pretrain_args['pc_num'],2048, 3),dtype=self.tensor_type).to(self.device)
                pc_aff_map_all = torch.ones((self.data_args['batch_size'],self.data_pretrain_args['pc_num'],2048, 3),dtype=self.tensor_type).to(self.device)


                (dynamic_kernels,           # [Batch_size, gs_embed_dim, 1]
                 pred_aff_map,              # [Batch_size, batch_max_points, 1]
                 text_loss,                 # tensor
                 predicted_text,            # list,[Batch_size]
                 ) = self.model(
                    gs_features, padded_gs_datas, masks, pc_mean_all, pc_aff_map_all, question, answer, device=self.device,use_csa=False)


                ce_loss = CE_loss(pred_aff_map,gs_aff_maps,masks)
                dice_loss = Dice_loss(pred_aff_map,gs_aff_maps,masks)
                batch_loss = ce_loss + dice_loss + text_loss


                batch_error,batch_points = MAE(pred_aff_map,gs_aff_maps,masks)  # [Batch size]，[Batch size]
                batch_sim = SIM(pred_aff_map,gs_aff_maps,masks)                 # [Batch_size]
                batch_kld = KLD(pred_aff_map,gs_aff_maps,masks)                 # [Batch_size]
                batch_auc = AUC(pred_aff_map, gs_aff_maps, masks)               # [Batch_size]
                batch_iou = IOU(pred_aff_map, gs_aff_maps, masks)               # [Batch_size]


                total_metrics['loss'] += batch_loss.item()
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
                                                    f'Batch {i + 1}/{len(self.dataloader_val)} | '
                                                    f'Loss: {batch_loss.item():.4f} | '
                                                    f'MAE: {batch_mae.item():.4f} | '
                                                    f'SIM: {torch.mean(batch_sim).item():.4f} | '
                                                    f'KLD: {torch.mean(batch_kld).item():.4f} | '
                                                    f'AUC: {torch.mean(batch_auc).item():.4f} | '
                                                    f'IOU: {torch.mean(batch_iou).item():.4f}')

            dist.barrier()
            metrics_tensor = torch.tensor([
                total_metrics['loss'],
                total_metrics['mae'],
                total_metrics['sim'],
                total_metrics['kld'],
                total_metrics['auc'],
                total_metrics['iou'],
                total_points,
                total_data
            ], dtype=self.tensor_type, device=self.device)


            all_metrics = [torch.zeros_like(metrics_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(all_metrics, metrics_tensor)


            if dist.get_rank() == 0:
                total_loss = 0.0
                total_mae = 0.0
                total_sim = 0.0
                total_kld = 0.0
                total_auc = 0.0
                total_iou = 0.0
                total_points = 0
                total_data = 0


                for metrics in all_metrics:
                    total_loss += metrics[0].item()
                    total_mae += metrics[1].item()
                    total_sim += metrics[2].item()
                    total_kld += metrics[3].item()
                    total_auc += metrics[4].item()
                    total_iou += metrics[5].item()
                    total_points += metrics[6].item()
                    total_data += metrics[7].item()


                total_metrics['loss'] = total_loss / total_data
                total_metrics['mae'] = total_mae / total_points
                total_metrics['sim'] = total_sim / total_data
                total_metrics['kld'] = total_kld / total_data
                total_metrics['auc'] = total_auc / total_data
                total_metrics['iou'] = total_iou / total_data


                self.log_with_time(eval_time,f'Epoch:{epoch} | Overall Evaluation Metrics')
                self.logger.info(f'  - Loss: {total_metrics["loss"]:.4f}')
                self.logger.info(f'  - MAE: {total_metrics["mae"]:.4f}')
                self.logger.info(f'  - SIM: {total_metrics["sim"]:.4f}')
                self.logger.info(f'  - KLD: {total_metrics["kld"]:.4f}')
                self.logger.info(f'  - AUC: {total_metrics["auc"]:.4f}')
                self.logger.info(f'  - IOU: {total_metrics["iou"]:.4f}')


                self.writer.add_scalar('Eval/Loss', total_metrics['loss'], epoch)
                self.writer.add_scalar('Eval/MAE', total_metrics['mae'], epoch)
                self.writer.add_scalar('Eval/SIM', total_metrics['sim'], epoch)
                self.writer.add_scalar('Eval/KLD', total_metrics['kld'], epoch)
                self.writer.add_scalar('Eval/AUC', total_metrics['auc'], epoch)
                self.writer.add_scalar('Eval/IOU', total_metrics['iou'], epoch)


                if total_metrics['iou'] > self.best_iou:
                    self.best_iou = total_metrics['iou']

                    merged_model = copy.deepcopy(self.model.module)
                    if isinstance(merged_model.mllm.model, PeftModel):
                        merged_model.mllm.model = merged_model.mllm.model.merge_and_unload()


                    save_dir = './checkpoints/pretrain/'+self.setting
                    os.makedirs(save_dir, exist_ok=True)

                    checkpoint_path = os.path.join(save_dir, f"pretrain_+"+self.setting+"_best.pth")
                    if os.path.exists(checkpoint_path):
                        self.logger.warning(f"Checkpoint {checkpoint_path} already exists, overwriting!")

                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': merged_model.state_dict(),
                        'optimizer_state_dict': self.pretrain_optimizer.state_dict(),
                        'scheduler_state_dict': self.pretrain_scheduler.state_dict()
                    }, checkpoint_path)

                    self.logger.info(f'Epoch:{epoch} | Saved best model with IOU: {self.best_iou:.4f}')

        dist.barrier()
        self.log_with_time(eval_time,f'————————————————EVALUATION END————————————————')


    def train(self):
        self.pretrain()
        self.writer.close()



if __name__ == '__main__':
    try:
        trainer = pretrain_trainer()
        trainer.train()
    except Exception as e:

        dist.destroy_process_group()
        raise


    dist.destroy_process_group()