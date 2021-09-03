import argparse
import numpy as np 
import os
import torch 
from utils import get_state_dict_from_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument('-source_path', type=str, default='', 
                    help='path to models whose kernel slice should be read out')
parser.add_argument('-target_path', type=str, default='',
                    help='where to unload the saved kernel slices ')
parser.add_argument('-arch', default='resnet20', 
                    help='which model is used to load kernel slices')
parser.add_argument('-D', type=int, default=3,
                    help='dimension of the kernel slices')
args = parser.parse_args()

k_dim = args.D 
dim_str = str(k_dim)+'x'+str(k_dim) #like 3x3 
target_path = os.path.join(args.target_path,dim_str)

for split in ['train','val']:
    full_source_path = os.path.join(args.source_path,split)
    slices = []
    for run in os.listdir(full_source_path):
        run_cp_path = os.path.join(full_source_path,run,'model.th')
        state_dict = get_state_dict_from_checkpoint(run_cp_path)
        layer = 0
        for param_tensor in state_dict: 
            if 'conv' in param_tensor: 
                params = state_dict[param_tensor]
                for 

                
        





