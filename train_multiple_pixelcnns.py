# first extract the latents from the models if not already existing
# for that we need: data_dir, model name to load the model
# save the latents there as ???? 
# then train a pixelcnn on that

import argparse
import enum
import os 
import utils
import torch
import numpy as np 
from utils import load_vqvae1
from my_utils import sorted_alphanumeric
from models import vqvae1,pixelcnn

def load_vqvae_model(vq_dir,vq_arch,args):
    if vq_arch.startswith('vqvae1'):
        vqvae = load_vqvae1(vq_dir,args.device)
    else:
        raise NotImplementedError('no {} init'.format(vq_arch))
    return vqvae

def extract_latents_from_model(layer_dir, arch, args):
    '''saves two numpy arrays containing all the latents of the train and testdata'''
    #load data 
    trainloader, _ = utils.get_dataloader(os.path.join(args.data_dir, 'train.npy'), args.batch_size, shuffle=False)
    testloader, _ = utils.get_dataloader(os.path.join(args.data_dir, 'test.npy'), args.test_bs, shuffle=False)

    #load model 
    vq_dir = os.path.join(layer_dir,arch)
    vqvae = load_vqvae_model(vq_dir,args)

    #get the latents by going over train and testloader
    train_latents = []
    test_latents = []
    for _,x in enumerate(trainloader):
        train_latents.append(vqvae._vq_vae(vqvae.encode(x))[-1].argmax(dim=1).view(-1,3,3))
    for _,x in enumerate(testloader):
        test_latents.append(vqvae._vq_vae(vqvae.encode(x))[-1].argmax(dim=1).view(-1,3,3))
    
    train_latents = torch.cat(train_latents)
    train_latents = train_latents.cpu().detach().numpy()
    test_latents = torch.cat(test_latents)
    test_latents = test_latents.cpu().detach().numpy()
    
    latent_dir = os.path.join(vq_dir,'latents')
    if not os.path.exists(latent_dir):
        os.makedirs(latent_dir)

    np.save(os.path.join(latent_dir,'train'),train_latents)
    np.save(os.path.join(latent_dir,'test'),test_latents)

def get_input_dim(args):
    vq_dir = os.path.join(args.data_root_path,'layer_0',
                        'vqvae{}.{}'.format(args.vqvae_arch,args.vqvae_spec))
    vqvae = load_vqvae_model(vq_dir,args)
    return vqvae.num_embeddings

def run_train_pixelcnn(args):
    string = 'python train-pixelcnn.py'
    for arg in vars(args):
        string += ' --'+str(arg)+' '+str(getattr(args,arg))
    os.system(string)

parser = argparse.ArgumentParser()
#general training arguments
parser.add_argument('--data_root_path', default=os.path.join('data','resnet20','3x3'), 
                    help='the dir where all layers are stored')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--gpu_id', default='0')
parser.add_argument('--vqvae_arch', type=str, 
                    help='''which vqvae architecture is used for getting latents
                    example: vqvae1''')
parser.add_argument('--vqvae_spec', type=str, help='''which paramter version is used 
                    for getting latents, example: 4 ''')

#for eval (outputs)
parser.add_argument('--eval_freq', default=1, type=int)
parser.add_argument('--test_bs', default=512, type=int)
parser.add_argument('--verbose', default=1, type=int)

#model spec 
parser.add_argument('--pixelcnn_spec', type=str, help='which paramter version of pixelcnn is this')
parser.add_argument('--z_dim', default=8, type=int)
parser.add_argument('--hidden_dim', default=16, type=int)
parser.add_argument('--kernel_dim', default=16, type=int)

# additional info
parser.add_argument('--start_at_layer', type=int, default=0, help='''at which layer the 
                        vqvaes should be started to train ''')

#introduced for later use 
parser.add_argument('--add_save_path',default='')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--input_dim',type=int, default=0,
                    help='''the number of encodings of the vqvae''')

args = parser.parse_args()

data_root_path = args.data_root_path
start_layer = args.start_at_layer
vq_arch = args.vqvae_arch
vq_spec = args.vqvae_spec 
pix_spec = args.pixelcnn_spec

#get the number of codebook vectors in the vq model
setattr(args,'input_dim',get_input_dim(args))

for layer in sorted_alphanumeric(os.listdir(data_root_path))[start_layer:]:
    #set and make paths 
    data_path = os.path.join(data_root_path,layer,
                        'vqvae{}.{}'.format(vq_arch,vq_spec),
                        'latents')
    pix_save_path = os.path.join(data_root_path,layer,
                        'vqvae{}.{}'.format(vq_arch,vq_spec),
                        'pixelcnn{}'.format(pix_spec))
    if not os.path.exists(pix_save_path):
        os.makedirs(pix_save_path)

    #change some args
    setattr(args,'data_dir',data_path)
    setattr(args, 'add_save_path',pix_save_path)

    #check whether data already exists, otherwise extract the latents
    extract_latents_from_model(os.path.join(data_root_path,layer),vq_arch,args)

    #delete attributes that arent used anymore
    delattr(args,'general_data_dir')
    delattr(args,'start_at_layer')
    delattr(args,'vqvae_arch')
    delattr(args,'vqvae_spec')
    delattr(args,'pixelcnn_spec')

    run_train_pixelcnn(args)