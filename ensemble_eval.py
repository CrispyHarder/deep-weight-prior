import argparse
import torch
import os 
from utils import load_cifar_c_loader,load_pcam_loaders,load_dataset
from my_utils import Ensemble,brier_multi,expected_calibration_error
import numpy as np 
from torch.nn import functional as F
import json 

def predict(data, net):
    prob = []
    pred = []
    l = []
    for x, y in data:
        l.append(y.numpy())
        x = x.to(device)
        p = net.predict(x)
        sp = F.softmax(p, dim=1)
        pred.append(p.data.cpu().numpy())
        prob.append(sp.data.cpu().numpy())
    return np.concatenate(pred), np.concatenate(prob), np.concatenate(l)

def eval_run(testloader,ensemble,corr_lvl,args):
    # get predictions and labels 
    logits, probs, labels = predict(testloader,ensemble)
    #get array of actualy predictions from the logits
    predictions = np.argmax(probs,axis=1)
    # get the confidences of each prediction = percentage of prediction
    confidences = [probs[i,int(predictions[i])] for i in range(len(logits))] 

    accuracy = np.sum(labels == predictions)/len(labels)
    ece = expected_calibration_error(confidences,predictions,labels)
    nll = float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)))
    brier = float(brier_multi(labels,probs))

    #prepare dict to save results
    results = {
        'dataset': args.data,
        'corr_lvl':corr_lvl,
        'init': args.init,
        'n_members': args.n_members,
        'scores':{
            'accuracy':accuracy,
            'ece':ece,
            'nll':nll,
            'brier':brier
        }
    }
    
    print(results)

    save_dir = os.path.join('logs','ensemble results',args.data)
    save_path = os.path.join(save_dir,f'{args.init} {args.n_members} {corr_lvl}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_path, 'w') as fp:
        json.dump(results, fp)

#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data',choices=['cifar10','cifarC','pcam'])
parser.add_argument('--init',choices=['vae','ghn_base','ghn_noise'])
parser.add_argument('--n_members',type=int,choices=[5,10])
parser.add_argument('--gpu_id',type=int,choices=[0,1,2,3,4,5,6,7])
args = parser.parse_args()

#set device
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#get ensemble 
ensemble = Ensemble(args.init,args.n_members,device)

#get dataloader and make run 
if args.data == 'cifar10':
    _,testloader = load_dataset('cifar',500,500)
    eval_run(testloader,ensemble,0,args)
elif args.data == 'pcam':
    _,_,testloader = load_pcam_loaders(500,500)
    eval_run(testloader,ensemble,0,args)
elif args.data == 'cifarC':
    for corr_level in [1,2,3,4,5]:
        testloader = load_cifar_c_loader(corr_level)
        eval_run(testloader,ensemble,corr_level,args)





