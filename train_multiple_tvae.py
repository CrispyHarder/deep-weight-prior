## Wrapper script to use the structure of the parser in train-vqvae1.py
import argparse
import os 
from my_utils import sorted_alphanumeric

def run_train_tvae(args):
    string = 'python train-tvae.py'
    for arg in vars(args):
        string += ' --'+str(arg)+' '+str(getattr(args,arg))
    os.system(string)

parser = argparse.ArgumentParser()

#train settings 
parser.add_argument('--data_dir', 
                    default=os.path.join('data','resnet20','3x3'))
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--gpu_id', default='0')

#optimisation
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--sgd_momentum',type=float, default=0.9)
parser.add_argument('--lr_decay_step', default=int(11e8), type=int)
parser.add_argument('--lr_decay', default=0.5, type=float)

#evaluation
parser.add_argument('--eval_freq', default=1, type=int)
parser.add_argument('--test_bs', default=512, type=int)
parser.add_argument('--verbose', default=1, type=int)

#model specifics
parser.add_argument('--tvae_spec', type=str, help='which paramter version')
parser.add_argument('--n_caps', default=5, type=int)
parser.add_argument('--cap_dim', default=4, type=int)
parser.add_argument('--mu_init', default=30, type=int)

#misc
parser.add_argument('--add_save_path',default='')
parser.add_argument('--start_at_layer', type=int, default=0, help='''at which layer the 
                        vqvaes should be started to train ''')

args = parser.parse_args()
data_root_path = args.data_dir
start_layer = args.start_at_layer
spec = args.tvae_spec 

delattr(args,'start_at_layer')
delattr(args,'tvae_spec')

for layer in sorted_alphanumeric(os.listdir(data_root_path))[start_layer:]:
    data_path = os.path.join(data_root_path,layer,'conv')
    tvae_save_path = os.path.join(data_root_path,layer,
                        'tvae{}'.format(spec))
    if not os.path.exists(tvae_save_path):
        os.makedirs(tvae_save_path)
    setattr(args,'data_dir',data_path)
    setattr(args, 'add_save_path',tvae_save_path)
    run_train_tvae(args)

