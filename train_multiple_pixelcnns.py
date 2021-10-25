# first extract the latents from the models if not already existing
# for that we need: data_dir, model name to load the model
# save the latents there as ???? 
# then train a pixelcnn on that

import argparse
import os 
from my_utils import sorted_alphanumeric
from models import vqvae1,pixelcnn

def extract_latents_from_model(layer_dir, arch):
    
    pass

def run_train_vqvae(args,spec):
    string = 'python train-vqvae{}.py'.format(spec)
    for arg in vars(args):
        string += ' --'+str(arg)+' '+str(getattr(args,arg))
    os.system(string)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='', 
                    help='the dir where all layers are stored')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--eval_freq', default=1, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--test_bs', default=512, type=int)
parser.add_argument('--z_dim', default=8, type=int)
parser.add_argument('--hidden_dim', default=16, type=int)
parser.add_argument('--kernel_dim', default=16, type=int)
parser.add_argument('--verbose', default=1, type=int)
parser.add_argument('--gpu_id', default='0')
parser.add_argument('--add_save_path',default='')
parser.add_argument('--vqvae_arch', type=str, help='which vqvae architecture is used')
parser.add_argument('--vqvae_spec', type=str, help='which paramter version')
parser.add_argument('--start_at_layer', type=int, default=0, help='''at which layer the 
                        vqvaes should be started to train ''')

args = parser.parse_args()
data_root_path = args.data_dir
start_layer = args.start_at_layer
arch = args.vqvae_arch
spec = args.vqvae_spec 

delattr(args,'start_at_layer')
delattr(args,'vqvae_arch')
delattr(args,'vqvae_spec')

for layer in sorted_alphanumeric(os.listdir(data_root_path))[start_layer:]:
    data_path = os.path.join(data_root_path,layer,'conv')
    vqvae_save_path = os.path.join(data_root_path,layer,
                        'vqvae{}.{}'.format(arch,spec))
    if not os.path.exists(vqvae_save_path):
        os.makedirs(vqvae_save_path)
    setattr(args,'data_dir',data_path)
    setattr(args, 'add_save_path',vqvae_save_path)
    run_train_vqvae(args,arch)