## Wrapper script to use the structure of the parser in train-vae.py
import argparse
import os 
from my_utils import sorted_alphanumeric

def get_nr_epochs(layer_nr):
    layer_nr = int(layer_nr.split('_')[1])
    if layer_nr == 0:
        return 150
    elif layer_nr <= 6:
        return 45
    elif layer_nr == 7:
        return 25
    elif layer_nr <= 12:
        return 10
    elif layer_nr == 13:
        return 7
    elif layer_nr <= 18: 
        return 4
    else:
        return RuntimeError
        
def run_train_vqvae(args):
    string = 'python train-vqvaeV2.py'
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
parser.add_argument('--weight_decay', type=float, default=1)

#evaluation
parser.add_argument('--test_bs', default=512, type=int)

#model specifics
parser.add_argument('--vqvae_spec', type=str, help='which paramter version')

parser.add_argument('--z_dim', default=8, type=int)
parser.add_argument('--hidden_dim', default=16, type=int)
parser.add_argument('--num_embeddings', type=int, default=128)
parser.add_argument('--ema_decay', type= float, default=0.)
parser.add_argument('--commitment_cost', type=float, default=0.25)

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
spec = args.vqvae_spec 

delattr(args,'start_at_layer')
delattr(args,'end_at_layer')
delattr(args,'dataset')
delattr(args,'vqvae_spec')

for layer in sorted_alphanumeric(os.listdir(data_root_path))[start_layer:end_layer+1]:
    data_path = os.path.join(data_root_path,layer,'conv')
    vqvae_save_path = os.path.join(data_root_path,layer,
                        f'vqvae{spec}')
    if not os.path.exists(vqvae_save_path):
        os.makedirs(vqvae_save_path)
    setattr(args,'data_dir',data_path)  
    setattr(args, 'add_save_path',vqvae_save_path)
    setattr(args, 'num_epochs',get_nr_epochs(layer))
    run_train_vqvae(args)

