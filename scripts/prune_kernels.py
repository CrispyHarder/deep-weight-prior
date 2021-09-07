import argparse
import numpy as np 
import os

parser = argparse.ArgumentParser()
parser.add_argument('-root', type=str, default='data',
                    help='where to unload the saved kernel slices ')
parser.add_argument('-arch', type=str, default='resnet20',
                    help='where to unload the saved kernel slices ')
parser.add_argument('-D', type=str, default='3',
                    help='where to unload the saved kernel slices ')
parser.add_argument('-percent', type=int, default=5,
                    help='slices with lowest x percent are pruned away')

args = parser.parse_args()

k_dim = args.D 
dim_str = str(k_dim)+'x'+str(k_dim)
path_to_convs = os.path.join(args.root,args.arch,dim_str)

for layer in os.listdir(path_to_convs):
    for split in ['train','test']:
        p_conv = os.path.join(path_to_convs,layer,'conv',split+'.npy')
        conv = np.load(p_conv)
        conv_norms = np.array([np.linalg.norm(slice) for slice in conv])
        threshold = np.percentile(conv_norms,args.percent)
        pruned_conv = np.array([s for s in conv if np.linalg.norm(s) >= threshold])
        np.save(p_conv,pruned_conv)

       