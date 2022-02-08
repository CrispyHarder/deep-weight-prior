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
        p = net(x)
        sp = F.softmax(p, dim=1)
        pred.append(p.data.cpu().numpy())
        prob.append(sp.data.cpu().numpy())
    return np.concatenate(pred), np.concatenate(prob), np.concatenate(l)


#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data',choices=['cifar10,cifarC,pcam'])
parser.add_argument('--corr_lvl',type=int,
                    help='''If the dataset  is cifarC, this is the corruption level,
                    otherwise this argument is not used''',choices=[1,2,3,4,5])
parser.add_argument('--init',choices=['vae','ghn_base','ghn_noise'])
parser.add_argument('--n_members',choices=[5,10])
parser.add_argument('--gpu_id',choices=[0,1,2,3,4,5,6,7])
args = parser.parse_args()

#set device
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#get dataloader
if args.data == 'cifar10':
    _,testloader = load_dataset('cifar',500,500)
elif args.data == 'pcam':
    _,_,testloader = load_pcam_loaders(500,500)
elif args.data == 'cifarC':
    testloader = load_cifar_c_loader(args.corr_lvl)

#get ensemble 
ensemble = Ensemble(args.init,args.n_members,device)

# get predictions and labels 
logits, probs, labels = predict(testloader,ensemble)
#get array of actualy predictions from the logits
predictions = logits.max(1)[1] 
# get the confidences of each prediction = percentage of prediction
confidences = [probs[i,predictions[i]] for i in range(len(logits))] 

accuracy = np.sum(labels == predictions)
ece = expected_calibration_error(confidences,predictions,labels)
nll = F.cross_entropy(logits, labels)
brier = brier_multi(labels,probs)

#prepare dict to save results
corr_lvl = 0 if not args.data == 'cifarC' else args.corr_lvl
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

save_dir = os.path.join('logs','ensemble results',args.data)
save_path = os.path.join(save_dir,f'{args.init} {args.n_members} {corr_lvl}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(save_dir, 'w') as fp:
    json.dump(results, fp)


