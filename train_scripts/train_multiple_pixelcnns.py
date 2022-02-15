# first extract the latents from the models if not already existing
# for that we need: data_dir, model name to load the model
# save the latents there as ???? 
# then train a pixelcnn on that

import argparse
import os 
import utils
import torch
import numpy as np 
from utils import load_vqvae1
from my_utils import sorted_alphanumeric

def load_vqvae_model(vq_dir,vq_name,device):
    if vq_name.startswith('vqvae1'):
        vqvae = load_vqvae1(vq_dir,device)
    else:
        raise NotImplementedError('no {} init'.format(vq_arch))
    return vqvae

def extract_latents_from_model(layer_dir, vq_name, slice_dir, device, args):
    '''saves two numpy arrays containing all the latents of the train and testdata'''
    #load data 
    trainloader, _ = utils.get_dataloader(os.path.join(slice_dir, 'train.npy'), args.batch_size, shuffle=False)
    testloader, _ = utils.get_dataloader(os.path.join(slice_dir, 'test.npy'), args.test_bs, shuffle=False)

    #load model 
    vq_dir = os.path.join(layer_dir,vq_name)
    vqvae = load_vqvae_model(vq_dir,vq_name,device)

    #get the latents by going over train and testloader
    train_latents = []
    test_latents = []
    for _,x in enumerate(trainloader):
        x = x.to(device)
        train_latents.append(vqvae._vq_vae(vqvae.encode(x))[-1].argmax(dim=1).view(-1,3,3))
    for _,x in enumerate(testloader):
        x = x.to(device)
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

def get_input_dim(device,vq_name):
    vq_dir = os.path.join(args.data_root_path,'layer_0',vq_name)
    vqvae = load_vqvae_model(vq_dir,vq_name,device)
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
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--num_epochs', default=100, type=int)
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
parser.add_argument('--dim', default=64, type=int,
                    help='''the hidden size (number of channels per layer)''')
parser.add_argument('--n_layers', type=int, default=10)

# additional info
parser.add_argument('--start_at_layer', type=int, default=0, help='''at which layer the 
                        pixelcnns should be started to train ''')

#introduced for later use 
parser.add_argument('--add_save_path',default='')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--input_dim',type=int, default=0,
                    help='''the number of encodings of the vqvae''')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_root_path = args.data_root_path
start_layer = args.start_at_layer
vq_arch = args.vqvae_arch
vq_spec = args.vqvae_spec 
pix_spec = args.pixelcnn_spec
vq_name = vq_arch+'.'+vq_spec
#get the number of codebook vectors in the vq model
setattr(args,'input_dim',get_input_dim(device,vq_name))

for i,layer in enumerate(sorted_alphanumeric(os.listdir(data_root_path))[start_layer:]):
    #set and make paths 
    data_path = os.path.join(data_root_path,layer,vq_name,'latents')
    pix_save_path = os.path.join(data_root_path,layer,vq_name,'pixelcnn{}'.format(pix_spec))
    if not os.path.exists(pix_save_path):
        os.makedirs(pix_save_path)

    #check whether data already exists, otherwise extract the latents
    slice_dir = os.path.join(data_root_path,layer,'conv')
    if not os.path.exists(data_path):
        extract_latents_from_model(os.path.join(data_root_path,layer),vq_name,slice_dir,device,args)

    #change some args
    setattr(args,'data_dir',data_path)
    setattr(args, 'add_save_path',pix_save_path)

    if i == 0:
        #delete attributes that arent used anymore
        delattr(args,'data_root_path')
        delattr(args,'start_at_layer')
        delattr(args,'vqvae_arch')
        delattr(args,'vqvae_spec')
        delattr(args,'pixelcnn_spec')

    run_train_pixelcnn(args)