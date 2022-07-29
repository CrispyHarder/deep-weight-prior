import os 
import torch 
from models.cifar import ResNet
import argparse
from utils import load_cifar10_loaders
from utils import load_pcam_dataloaders
import numpy as np 
from torch.nn import functional as F
import json 

def predict(data, net):
    pred = []
    l = []
    for x, y in data:
        l.append(y.numpy())
        x = x.to(device)
        p = F.log_softmax(net(x), dim=1)
        pred.append(p.data.cpu().numpy())
    return np.concatenate(pred), np.concatenate(l)

def get_perf(data,net):
    pred, labels = predict(data,net)
    acc = np.mean(pred.argmax(1) == labels)
    nll = -pred[np.arange(len(labels)), labels].mean()
    return acc,nll
    
device = torch.device('cuda:1') 
DATASETS = ['cifar','pcam']


# _,_,cifar_loader = load_cifar10_loaders(50,500)
_,_,pcam_loader = load_pcam_dataloaders(64)


# ### CIFAR ### 
# model = ResNet([3,3,3],num_classes=10).to(device)
# path = os.path.join('logs',f'exman-train-net-cifar.py','runs')
# for run in os.listdir(path):
#     sd_path = os.path.join(path,run,'net_params.torch')
#     sd = torch.load(sd_path,map_location=device)
#     model.load_state_dict(sd)
#     acc,nll = get_perf(cifar_loader,model)
#     nll = np.float64(nll)
#     perf_dict = {'acc':acc,'nll':nll}
#     with open(os.path.join(path,run,'perf_dict.json'),'w') as fp:
#         json.dump(perf_dict,fp)


### PCAM ### 
model = ResNet([3,3,3],num_classes=2).to(device)
path = os.path.join('logs',f'exman-train-net-pcam.py','runs')
for run in os.listdir(path):
    sd_path = os.path.join(path,run,'net_params.torch')
    sd = torch.load(sd_path,map_location=device)
    model.load_state_dict(sd)
    with torch.no_grad():
        acc,nll = get_perf(pcam_loader,model)
    nll = np.float64(nll)
    perf_dict = {'acc':acc,'nll':nll}
    with open(os.path.join(path,run,'perf_dict.json'),'w') as fp:
        json.dump(perf_dict,fp)

    





