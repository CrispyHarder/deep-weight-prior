## Wrapper script to use the structure of the parser in train-vqvae1.py
import argparse
import os 
from my_utils import sorted_alphanumeric

def run_train_vqvae(args,arch):
    string = 'python train-vqvae{}.py'.format(arch)
    for arg in vars(args):
        string += ' --'+str(arg)+' '+str(getattr(args,arg))
    os.system(string)

parser = argparse.ArgumentParser()
parser.add_argument('--train')
parser.add_argument('--test')
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
parser.add_argument('--lr_decay_step', default=int(11e8), type=int)
parser.add_argument('--decay', default=0.5, type=float)
parser.add_argument('--var', default='train')
parser.add_argument('--add_save_path',default='')
parser.add_argument('--vqvae_arch', type=str, help='which vqvae architecture is used')
parser.add_argument('--vqvae_spec', type=str, help='which paramter version')
parser.add_argument('--num_embeddings', type=int, default=128)
parser.add_argument('--ema_decay', type= float, default=0.)
parser.add_argument('--commitment_cost', type=float, default=0.25)
parser.add_argument('--weight_decay', type=float, default=0.)
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

