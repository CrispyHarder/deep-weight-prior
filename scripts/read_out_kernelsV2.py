import argparse
import numpy as np 
import os
import torch 
from my_utils import load_statedict


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='', 
                    help='path to models whose kernel slice should be read out',
                    choices=['cifar','pcam'])

args = parser.parse_args()

target_path = os.path.join('data',f'resnet20_{args.dataset}','3x3')
source_path = os.path.join('logs',f'exman-train-net-{args.dataset}.py','runs')

slices = [[] for _ in range(19)]
for run in os.listdir(source_path)[:80]:
    run_cp_path = os.path.join(source_path,run,'net_params_lastepoch.torch')
    state_dict = load_statedict(run_cp_path,map_location=torch.device('cpu'))
    layer = 0
    for param_tensor in state_dict: 
        if 'conv' in param_tensor: 
            params = state_dict[param_tensor]
            p_shape = params.shape 
            slices[layer].append(torch.reshape(params,(p_shape[0]*p_shape[1],p_shape[2],p_shape[3])))
            layer += 1 
cat_slices = [torch.cat(entry) for entry in slices]
np_slices = [np.expand_dims(t.numpy(),axis=1) for t in cat_slices]
for n_layer in range(19):
    full_target_path = os.path.join(target_path,'layer_'+str(n_layer),'conv')
    if not os.path.exists(full_target_path):
        os.makedirs(full_target_path)
    np.save(os.path.join(full_target_path,'train'),np_slices[n_layer])

slices = [[] for _ in range(19)]
for run in os.listdir(source_path)[80:]:
    run_cp_path = os.path.join(source_path,run,'net_params_lastepoch.torch')
    state_dict = load_statedict(run_cp_path,map_location=torch.device('cpu'))
    layer = 0
    for param_tensor in state_dict: 
        if 'conv' in param_tensor: 
            params = state_dict[param_tensor]
            p_shape = params.shape 
            slices[layer].append(torch.reshape(params,(p_shape[0]*p_shape[1],p_shape[2],p_shape[3])))
            layer += 1 
cat_slices = [torch.cat(entry) for entry in slices]
np_slices = [np.expand_dims(t.numpy(),axis=1) for t in cat_slices]
for n_layer in range(19):
    full_target_path = os.path.join(target_path,'layer_'+str(n_layer),'conv')
    if not os.path.exists(full_target_path):
        os.makedirs(full_target_path)
    np.save(os.path.join(full_target_path,'test'),np_slices[n_layer])