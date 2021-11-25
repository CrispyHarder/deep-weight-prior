## Wrapper script to use the structure of the parser in train-vqvae1.py
import argparse
import os 
from my_utils import sorted_alphanumeric

def run_train_lvae(args):
    string = 'python train-lvae.py'
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

#evaluation
parser.add_argument('--eval_freq', default=1, type=int)
parser.add_argument('--test_bs', default=512, type=int)
parser.add_argument('--verbose', default=1, type=int)

#model specifics
parser.add_argument('--lvae_spec', type=str, help='which paramter version')
parser.add_argument('--dims', default=[100,50,10], nargs='*', type=float)
parser.add_argument('--z_dim', default=2, type=int)

#misc
parser.add_argument('--add_save_path',default='')
parser.add_argument('--start_at_layer', type=int, default=0, help='''at which layer the 
                        vqvaes should be started to train ''')
parser.add_argument('--end_at_layer', type=int, default=0, help='''at which layer the 
                        vqvaes should be started to train ''')

args = parser.parse_args()
data_root_path = args.data_dir
start_layer = args.start_at_layer
end_layer = args.end_at_layer
spec = args.lvae_spec 

delattr(args,'start_at_layer')
delattr(args,'end_at_layer')
delattr(args,'lvae_spec')

for layer in sorted_alphanumeric(os.listdir(data_root_path))[start_layer:end_layer+1]:
    data_path = os.path.join(data_root_path,layer,'conv')
    lvae_save_path = os.path.join(data_root_path,layer,
                        'lvae{}'.format(spec))
    if not os.path.exists(lvae_save_path):
        os.makedirs(lvae_save_path)
    setattr(args,'data_dir',data_path)
    setattr(args, 'add_save_path',lvae_save_path)
    run_train_lvae(args)

