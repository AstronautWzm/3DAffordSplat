import numpy as np
from scipy.spatial import KDTree
import torch
from scipy.spatial import distance
from geomloss import SamplesLoss


# Calculate the shape similarity between the Gaussian model and the point cloud model within a set of AffordSplat as the weight for the subsequent loss
'''
    Input:
        gs_means: ndarray, [batch_size, points_num, 3(xyz)]   
        pc_mean_all: ndarray, [batch_size, pc_num, points_num, 3(xyz)] 
    Output:
        loss1_weight: ndarray, [batch_size, pc_num]
'''
def gpse(gs_means,pc_mean_all,mask,T=np.float32(0.1),type="Chamfer"):
    batch_size = gs_means.shape[0]
    batch_max_points = gs_means.shape[1]


    gs_means_valid = []
    for batch_id in range(batch_size):
        valid_points = gs_means[batch_id][mask[batch_id] == 1]
        gs_means_valid.append(valid_points.reshape(-1, 3))  # [valid_points_num, 3]


    pc_mean_all=pc_rotation(pc_mean_all)


    if type=="Chamfer":
        dis_all_batch=Chamfer_dis(gs_means,pc_mean_all)
    elif type=="Hausdorff":
        dis_all_batch=Hausdorff_dis(gs_means,pc_mean_all)
    elif type=="Wasserstein":
        dis_all_batch=Wasserstein_dis(gs_means,pc_mean_all)
    else:
        raise Exception('gpse type not supported')


    dis_all_batch=[[-dis for dis in dis_all] for dis_all in dis_all_batch]
    dis_all_batch=dis_all_batch / T
    exp_dis=np.exp(dis_all_batch)
    exp_dis_sum=np.sum(exp_dis,axis=1,keepdims=True)
    loss1_weight=exp_dis/exp_dis_sum
    loss1_weight=np.array(loss1_weight)
    return loss1_weight


# Chamfer
'''
    Input:
        gs_means: ndarray, [batch_size, points_num, 3(xyz)]   
        pc_mean_all: ndarray, [batch_size, pc_num, points_num, C(xyz)] 
    Output:
        dis_all_batch: list, [batch_size, pc_num]
'''
def Chamfer_dis(gs_means,pc_mean_all):
    batch_size=gs_means.shape[0]
    pc_num=pc_mean_all.shape[1]
    dis_all_batch=[]

    for batch_id in range(batch_size):
        dis_all=[]
        for pc_id in range(pc_num):
            tree_gs = KDTree(gs_means[batch_id])
            tree_pc = KDTree(pc_mean_all[batch_id][pc_id])
            dist_gs = tree_pc.query(gs_means[batch_id])[0]
            dist_pc = tree_gs.query(pc_mean_all[batch_id][pc_id])[0]
            chamfer_dist = np.mean(dist_gs) + np.mean(dist_pc)
            dis_all.append(chamfer_dist)
        dis_all_batch.append(dis_all)

    return dis_all_batch


# Hausdorff
def Hausdorff_dis(gs_means,pc_mean_all):
    batch_size = gs_means.shape[0]
    pc_num=pc_mean_all.shape[1]
    dis_all_batch=[]
    for batch_id in range(batch_size):
        dis_all=[]
        for pc_id in range(pc_num):

            u= distance.directed_hausdorff(gs_means[batch_id],pc_mean_all[batch_id][pc_id])[0]
            v= distance.directed_hausdorff(pc_mean_all[batch_id][pc_id],gs_means[batch_id])[0]
            hausdorff_dist = max(u, v)
            dis_all.append(hausdorff_dist)
        dis_all_batch.append(dis_all)
    return dis_all_batch


# Wasserstein
def Wasserstein_dis(gs_means,pc_mean_all,reg=1.0):
    batch_size = gs_means.shape[0]
    pc_num = pc_mean_all.shape[1]
    distances = []

    # Define the Wasserstein distance function using geomloss
    loss = SamplesLoss("sinkhorn", p=2, blur=reg)

    for gs, pc_list in zip(gs_means, pc_mean_all):
        batch_dists = []
        for pc in pc_list:
            # Convert to PyTorch tensors
            gs_tensor = torch.from_numpy(gs).float()
            pc_tensor = torch.from_numpy(pc).float()

            # Compute Wasserstein distance
            dist = loss(gs_tensor, pc_tensor).item()
            batch_dists.append(dist)
        distances.append(batch_dists)
    return dis_all_batch



def pc_rotation(pc_mean_all):

    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    rotation_matrix_z = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])

    pc_mean_all = np.dot(pc_mean_all, rotation_matrix_x)
    pc_mean_all = np.dot(pc_mean_all, rotation_matrix_z)

    return pc_mean_all