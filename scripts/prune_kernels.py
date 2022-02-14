import argparse
import numpy as np 
import os

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str)
parser.add_argument('-percent', type=int, default=5,
                    help='slices with lowest x percent are pruned away')
                    
args = parser.parse_args()

source_dataset = os.path.join('data',f'resnet20_{args.dataset}','3x3')

for layer in os.listdir(source_dataset):
    for split in ['train','test']:
        p_conv = os.path.join(source_dataset,layer,'conv',split+'.npy')
        conv = np.load(p_conv)
        conv_norms = np.array([np.linalg.norm(slice) for slice in conv])
        threshold = np.percentile(conv_norms,args.percent)
        pruned_conv = np.array([s for s in conv if np.linalg.norm(s) >= threshold])
        np.save(p_conv,pruned_conv)

       