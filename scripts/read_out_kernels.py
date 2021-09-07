import argparse
import numpy as np 
import os
import torch 
from my_utils import get_state_dict_from_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument('-source_path', type=str, default='', 
                    help='path to models whose kernel slice should be read out')
parser.add_argument('-target_path', type=str, default='',
                    help='where to unload the saved kernel slices ')
parser.add_argument('-arch', default='resnet20', 
                    help='which model is used to load kernel slices')
parser.add_argument('-D', type=int, default=3,
                    help='dimension of the kernel slices')
parser.add_argument('-n_conv_layers', type=int, help='number of conv layers in the net')

args = parser.parse_args()

k_dim = args.D 
dim_str = str(k_dim)+'x'+str(k_dim) #like 3x3 
target_path = os.path.join(args.target_path,args.arch,dim_str)

for split in ['train','val']:
    full_source_path = os.path.join(args.source_path,split)
    if split == 'val':
        split = 'test' 
    slices = [[] for _ in range(args.n_conv_layers)]
    for run in os.listdir(full_source_path):
        run_cp_path = os.path.join(full_source_path,run,'model.th')
        state_dict = get_state_dict_from_checkpoint(run_cp_path,map_location=torch.device('cpu'))
        layer = 0
        for param_tensor in state_dict: 
            if 'conv' in param_tensor: 
                params = state_dict[param_tensor]
                p_shape = params.shape 
                slices[layer].append(torch.reshape(params,(p_shape[0]*p_shape[1],p_shape[2],p_shape[3])))
                layer += 1 
    cat_slices = [torch.cat(entry) for entry in slices]
    np_slices = [t.numpy() for t in cat_slices]
    for n_layer in range(args.n_conv_layers):
        full_target_path = os.path.join(target_path,'layer_'+str(n_layer),'conv')
        if not os.path.exists(full_target_path):
            os.makedirs(full_target_path)
        np.save(os.path.join(full_target_path,split),np_slices[n_layer])
        