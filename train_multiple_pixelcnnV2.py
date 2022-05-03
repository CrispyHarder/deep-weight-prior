## Wrapper script to use the structure of the parser in train-vae.py
import argparse
import os 
import utils
import torch
import numpy as np 
from utils import load_vqvae1
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
        
def load_vqvae_model(vq_dir,device):
    vqvae = load_vqvae1(vq_dir,device)
    return vqvae

def extract_latents_from_model(layer_dir, vq_name, slice_dir, device, args):
    '''saves two numpy arrays containing all the latents of the train and testdata'''
    #load data 
    trainloader, _ = utils.get_dataloader(os.path.join(slice_dir, 'train.npy'), args.batch_size, shuffle=False)
    testloader, _ = utils.get_dataloader(os.path.join(slice_dir, 'test.npy'), args.test_bs, shuffle=False)

    #load model 
    vq_dir = os.path.join(layer_dir,vq_name)
    vqvae = load_vqvae_model(vq_dir,device)

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
    global data_root_path
    vq_dir = os.path.join(data_root_path,'layer_0',vq_name)
    vqvae = load_vqvae_model(vq_dir,device)
    return vqvae.num_embeddings

def run_train_pixelcnnV2(args):
    string = 'python train-pixelcnnV2.py'
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
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--weight_decay', type=float, default=0.)

#evaluation
parser.add_argument('--test_bs', default=512, type=int)

#model specifics
parser.add_argument('--vqvae_spec', type=str, help='''which paramter version is used 
                    for getting latents, example: 4''', default=1)

parser.add_argument('--input_dim', default=8, type=int,
                        help='''the size of the codebook of the vqvae''') #is resetet later 
parser.add_argument('--dim', default=16, type=int,
                    help='''the hidden size (number of channels per layer)''')
parser.add_argument('--n_layers', type=int, default=10)

#misc
parser.add_argument('--start_at_layer', type=int, default=0, help='''at which layer the 
                        vqvaes should be started to train ''')
parser.add_argument('--end_at_layer', type=int, default=18, help='''after which layer the 
                        vqvaes should be stopped to train ''')

parser.add_argument('--add_save_path',default='')#for later use 

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_root_path = os.path.join('data',f'resnet20_{args.dataset}','3x3')
start_layer = args.start_at_layer
end_layer = args.end_at_layer
vq_spec = args.vqvae_spec 
vq_name = 'vqvae'+str(vq_spec)
#get the number of codebook vectors in   the vq model
setattr(args,'input_dim',get_input_dim(device,vq_name))

for i,layer in enumerate(sorted_alphanumeric(os.listdir(data_root_path))[start_layer:end_layer+1]):
    #set and make paths 
    data_path = os.path.join(data_root_path,layer,vq_name,'latents')
    pix_save_path = os.path.join(data_root_path,layer,vq_name,'pixelcnn')
    if not os.path.exists(pix_save_path):
        os.makedirs(pix_save_path)

    #check whether data already exists, otherwise extract the latents
    slice_dir = os.path.join(data_root_path,layer,'conv')
    if not os.path.exists(data_path):
        extract_latents_from_model(os.path.join(data_root_path,layer),vq_name,slice_dir,device,args)

    #change some args
    setattr(args,'data_dir',data_path)
    setattr(args, 'add_save_path',pix_save_path)
    setattr(args, 'num_epochs',get_nr_epochs(layer))

    if i == 0:
        #delete attributes that arent used anymore
        delattr(args,'start_at_layer')
        delattr(args,'end_at_layer')
        delattr(args,'dataset')     
        delattr(args,'vqvae_spec')

    run_train_pixelcnnV2(args)

