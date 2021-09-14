import torch 
from models.cifar import resnet
import yaml
import os 

def get_state_dict_from_checkpoint(checkpoint_path, map_location=None):
    '''loads the state dict from a given checkpoint path'''
    if map_location:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    elif torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    return checkpoint['state_dict']

def load_resnet_from_checkpoint(checkpoint_path, model_type, dataset_name):
    '''Gets a path to a checkpoint and a model type and loads the model 
    using the state dict'''

    model = resnet.__dict__[model_type]()
    model.load_state_dict(get_state_dict_from_checkpoint(checkpoint_path))
    if torch.cuda.is_available():
        model.cuda()  
    return model

def save_args_params(args, dir_path):
    dumbd = args.__dict__.copy()
    path = os.path.join(dir_path,'params.yaml')
    with open(path,'a') as f:
        yaml.dump(dumbd,f)
