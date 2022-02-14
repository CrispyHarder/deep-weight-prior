## Wrapper script to use the structure of the parser in train-vae.py
import argparse
import os 
from my_utils import sorted_alphanumeric

def run_train_vae(args):
    string = 'python train-vaeV2.py'
    for arg in vars(args):
        string += ' --'+str(arg)+' '+str(getattr(args,arg))
    os.system(string)

parser = argparse.ArgumentParser()

#train settings
parser.add_argument('--dataset', type=str)

parser.add_argument('--data_dir',default='') # for later use 
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--gpu_id', default='0')

#optimisation 
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', type=float, default=0.0001)

#evaluation
parser.add_argument('--test_bs', default=512, type=int)

#model specifics
parser.add_argument('--z_dim', default=2, type=int)
parser.add_argument('--hidden_dim', default=16, type=int)

#misc
parser.add_argument('--start_at_layer', type=int, default=0, help='''at which layer the 
                        vqvaes should be started to train ''')
parser.add_argument('--end_at_layer', type=int, default=18, help='''after which layer the 
                        vqvaes should be stopped to train ''')

parser.add_argument('--add_save_path',default='')#for later use 




args = parser.parse_args()

data_root_path = os.path.join('data',f'resnet20_{args.dataset}','3x3')
start_layer = args.start_at_layer
end_layer = args.end_at_layer

delattr(args,'start_at_layer')
delattr(args,'end_at_layer')
delattr(args,'dataset')

for layer in sorted_alphanumeric(os.listdir(data_root_path))[start_layer:end_layer+1]:
    data_path = os.path.join(data_root_path,layer,'conv')
    vae_save_path = os.path.join(data_root_path,layer,'vae')
    if not os.path.exists(vae_save_path):
        os.makedirs(vae_save_path)
    setattr(args,'data_dir',data_path)  
    setattr(args, 'add_save_path',vae_save_path)
    run_train_vae(args)

