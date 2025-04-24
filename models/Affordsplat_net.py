import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Pointnet_utils import Pointnet_with_transformer
from models.MMFM import MMFM,MLP
from models.GAM import *
from models.Decoder import Decoder
from models.MLLM import MLLM
from CSA import CSA
from torchsummary import summary

# Affordsplat_Net
'''
    Input:
        __init__:
            model_args: dict
        forward:
            gs_features: [batch size,sampled_points,C]
            padded_gs_datas:[batch size,batch_max_points,10]
            masks:[batch size,batch_max_points](二值掩码)
            pc_mean_all:[batch size,pc_num,points_num,3]
            pc_aff_map_all: [batch size,pc_num,points_num,1](概率掩码)
            question: [Batch size]
            answer: [Batch size]
    Output:
        forward:
            dynamic_kernels: [Batch_size, gs_embed_dim, 1]
            gs_aff_map: [Batch_size, batch_max_points, 1]
            text_loss: tensor
            predicted_text: list,[Batch_size,str]
            gs_stru_features: [Batch_size, nsample, embed_dim] 
            pc_stru_features: [Batch_size, pc_num, points_num, embed_dim]
'''
class Affordsplat_net(nn.Module):
    def __init__(self,model_args):
        super(Affordsplat_net,self).__init__()
        self.model_args = model_args
        att_drop = self.model_args['utils']['att_drop']
        lin_drop = self.model_args['utils']['lin_drop']
        num_heads = self.model_args['Pointnet_with_transformer']['num_heads']
        transformer_encoder_num = self.model_args['Pointnet_with_transformer']['transformer_encoder_num']


        # notice:
        # 1、sum(out_channels[i][-1])+3 == ffn_list[-1] and ffn_list[-1] % num_heads ==0
        # 2、dim_in == 10 or sum(out_channels[i][-1])+3+3
        self.point_encoder1 = Pointnet_with_transformer(self.model_args['Pointnet_with_transformer']['net1']['nsample'],
                                                        self.model_args['Pointnet_with_transformer']['net1']['knn_points'],
                                                        self.model_args['utils']['gs_dim'],
                                                        [[16, 16, 32], [32, 32, 61], [32, 32, 64]],
                                                        [160, 160],
                                                        num_heads,
                                                        att_drop = att_drop,
                                                        lin_drop = lin_drop,
                                                        transformer_encoder_num=transformer_encoder_num)

        self.point_encoder2 = Pointnet_with_transformer(self.model_args['Pointnet_with_transformer']['net2']['nsample'],
                                                        self.model_args['Pointnet_with_transformer']['net2']['knn_points'],
                                                        163,
                                                        [[32, 32, 64], [64, 64, 125], [64, 64, 128]],
                                                        [320, 320],
                                                        num_heads,
                                                        att_drop=att_drop,
                                                        lin_drop=lin_drop,
                                                        transformer_encoder_num=transformer_encoder_num)

        self.point_encoder3 = Pointnet_with_transformer(self.model_args['Pointnet_with_transformer']['net3']['nsample'],
                                                        self.model_args['Pointnet_with_transformer']['net3']['knn_points'],
                                                        323,
                                                        [[64, 64, 128], [128, 128, 253], [128, 128, 256]],
                                                        [640, 640],
                                                        num_heads,
                                                        att_drop=att_drop,
                                                        lin_drop=lin_drop,
                                                        transformer_encoder_num=transformer_encoder_num)


        gs_embed_dim = self.model_args['utils']['gs_embed_dim']
        self.mlp1 = MLP(160, [160, gs_embed_dim])
        self.mlp2 = MLP(320, [320, gs_embed_dim])
        self.mlp3 = MLP(640, [640, gs_embed_dim])


        self.mllm = MLLM(text_encoder_type = self.model_args['mllm']['text_encoder_type'],
                         freeze_model=self.model_args['mllm']['freeze_model'],
                         embedding_dim=self.model_args['mllm']['embedding_dim'])


        self.mmfm = MMFM([128, 256, 512],
                         dim_q = self.model_args['mllm']['embedding_dim'],
                         dim_kv=gs_embed_dim,
                         num_heads=self.model_args['mmfm']['num_heads'],
                         dim_out= self.model_args['mmfm']['dim_out'],
                         maxpoints = [self.model_args['Pointnet_with_transformer']['net1']['nsample'],
                                      self.model_args['Pointnet_with_transformer']['net2']['nsample'],
                                      self.model_args['Pointnet_with_transformer']['net3']['nsample']],
                         att_drop=att_drop,
                         lin_drop=lin_drop)


        nsamples_list = [self.model_args['Pointnet_with_transformer']['net1']['nsample'],
                         self.model_args['Pointnet_with_transformer']['net2']['nsample'],
                         self.model_args['Pointnet_with_transformer']['net3']['nsample']]
        self.gam = GAM(nsamples_list,
                       gs_embed_dim,
                       out_channels=self.model_args['gam']['out_channels'],
                       mlp_ratio= self.model_args['gam']['mlp_ratio'],
                       k = self.model_args['gam']['k'],
                       p = self.model_args['gam']['p'])


        self.upsample = Optimized_Upsample()


        self.decoder = Decoder(self.model_args['mllm']['embedding_dim'],
                               gs_embed_dim,
                               num_heads = self.model_args['decoder']['num_heads'],
                               decoder_block_num = self.model_args['decoder']['decoder_block_num'],
                               lin_after_decode=True)


        self.csa=CSA(self.model_args['utils']['gs_dim'],
                     self.model_args['utils']['pc_dim'],
                     embed_dim=self.model_args['csa']['embed_dim'],
                     num_heads=self.model_args['csa']['num_heads'],
                     nsample=self.model_args['csa']['nsample'],
                     knn_points=self.model_args['csa']['knn_points'],
                     mlp_list=[[32, 32, 64], [64, 64, 125], [64, 64, 128]],
                     att_drop = att_drop,
                     lin_drop = lin_drop)

    def forward(self,gs_features, padded_gs_datas, masks, pc_xyz_all, pc_aff_map_all, question, answer, device = 'cuda',use_csa=True):

        padded_gs_xyz=padded_gs_datas[:, :, :3] # [batch_size, batch_max_points, 3]


        features1, centroids_xyz1 = self.point_encoder1(gs_features)  # [batch_size, 1024, 160]
        features1_xyz = torch.cat((centroids_xyz1, features1), dim=-1)

        features2, centroids_xyz2 = self.point_encoder2(features1_xyz)  # [batch_size, 512, 320]
        features2_xyz = torch.cat((centroids_xyz2, features2), dim=-1)

        features3, centroids_xyz3 = self.point_encoder3(features2_xyz)  # [batch_size, 256, 640]


        features1 = self.mlp1(features1) # [batch_size, 1024, gs_embed_dim]
        features2 = self.mlp2(features2) # [batch_size, 512, gs_embed_dim]
        features3 = self.mlp3(features3) # [batch_size, 256, gs_embed_dim]


        text_encoded_dict = self.mllm(question, answer, device)
        aff_embedding = text_encoded_dict["aff_hidden_states"]
        aff_embedding.unsqueeze_(1)  # [batch_size, 1, embed_dim]
        text_loss = text_encoded_dict["text_loss"] # tensor
        predicted_text = text_encoded_dict['predicted_text']


        fused_feature1 = self.mmfm(aff_embedding, features1, features1)  # [batch_size, 1024, gs_embed_dim]
        fused_feature2 = self.mmfm(aff_embedding, features2, features2)  # [batch_size, 512, gs_embed_dim]
        fused_feature3 = self.mmfm(aff_embedding, features3, features3)  # [batch_size, 256, gs_embed_dim]


        fused_feature, weight = self.gam([fused_feature3, fused_feature2, fused_feature1],
                                         [centroids_xyz3, centroids_xyz2, centroids_xyz1])  # [batch_size, 1024, gs_embed_dim]; [batch_size, 3, gs_embed_dim]


        upsanmpled_fused_feature = self.upsample(padded_gs_xyz,
                                                 centroids_xyz1,
                                                 None,
                                                 fused_feature,
                                                 masks)  # [Batch_size, batch_max_points, gs_embed_dim]



        dynamic_kernels = self.decoder(aff_embedding, fused_feature)  # [Batch_size, 1, gs_embed_dim]
        dynamic_kernels = torch.transpose(dynamic_kernels, 2, 1)  # [Batch_size, gs_embed_dim, 1]


        upsanmpled_fused_feature = F.normalize(upsanmpled_fused_feature, p=2, dim=1) # [Batch_size, batch_max_points, gs_embed_dim]


        gs_aff_map = F.sigmoid(torch.bmm(upsanmpled_fused_feature,dynamic_kernels)) # [Batch_size, batch_max_points, 1]


        gs_aff_map = gs_aff_map * masks.unsqueeze(-1) # [Batch_size, batch_max_points, 1]

        if use_csa:
            gs_stru_features, pc_stru_features = self.csa(padded_gs_datas,
                                                          gs_aff_map,
                                                          pc_xyz_all,
                                                          pc_aff_map_all,
                                                          masks,
                                                          threshold=self.model_args['csa']['threshold']) # [Batch_size, nsampled, embed_dim] ; [Batch_size, pc_num, points_num, embed_dim]

            return dynamic_kernels, gs_aff_map, text_loss, predicted_text, gs_stru_features, pc_stru_features
        else:
            return dynamic_kernels, gs_aff_map, text_loss, predicted_text



    def print_module_params(self):
        for name, module in self.named_children():
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"Module: {name}, Parameters: {param_count}")



    def freeze_modules(self, freeze_module=None):
        for name, module in self.named_children():
            if name in freeze_module:
                for param in module.parameters():
                    param.requires_grad = False


