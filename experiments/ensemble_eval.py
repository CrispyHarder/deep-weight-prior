import argparse
import torch
import os 
from utils import load_cifar10_loaders, load_cifar_c_loader
from my_utils import Ensemble,brier_multi,expected_calibration_error
import numpy as np 
from torch.nn import functional as F
import json 
import pandas as pd 
import pickle 
INIT_NAMES = [['vae'],['he'],['xavier'],['vqvae1'],['vqvae1','pixelcnn'],['tvae'],['lvae'],['ghn_base'],['ghn_loss'],['ghn_ce']]

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
def eval_run(ensemble,corr_lvl,init,n_members,result_list):
    '''gets a dataloader and an ensemble with n members 
    and produces 20 samples of that constellation, that are added 
    to a given list'''
    # get predictions and labels 
    for i in range(20):
        with torch.no_grad():
            logits, probs, labels = ensemble.get_ens_prediction(n_members,corr_level)
        #get array of actualy predictions from the logits
        predictions = np.argmax(probs,axis=1)
        # get the confidences of each prediction = percentage of prediction
        confidences = [probs[i,int(predictions[i])] for i in range(len(logits))] 

        accuracy = np.sum(labels == predictions)/len(labels)
        ece = expected_calibration_error(confidences,predictions,labels)
        nll = float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)))
        brier = float(brier_multi(labels,probs))
        result_list.append([corr_lvl,init,n_members,accuracy,ece,nll,brier])

    # #prepare dict to save results
    # results = {
    #     'dataset': args.data,
    #     'corr_lvl':corr_lvl,
    #     'init': args.init,
    #     'n_members': args.n_members,
    #     'scores':{
    #         'accuracy':accuracy,
    #         'ece':ece,
    #         'nll':nll,
    #         'brier':brier
    #     }
    # }
    
    # print(results)

    # save_dir = os.path.join('logs','ensemble_results')
    # save_path = os.path.join(save_dir,f'{corr_lvl} {args.init} {args.n_members}')
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # with open(save_path, 'w') as fp:
    #     json.dump(results, fp)

#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id',type=int,choices=[0,1,2,3,4,5,6,7])
args = parser.parse_args()

#set device
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_,_,c0 = load_cifar10_loaders(64,500)
c1 = load_cifar_c_loader(1)
c2 = load_cifar_c_loader(2)
c3 = load_cifar_c_loader(3)
c4 = load_cifar_c_loader(4)
c5 = load_cifar_c_loader(5)

dataloaders = [c0,c1,c2,c3,c4,c5]


result_list = [] 
for init in INIT_NAMES:
    ensemble = Ensemble(init,device,dataloaders)
    for n_members in [5,10]:
        print(n_members)
        for corr_level in [0,1,2,3,4,5]:
            print(corr_level)
            eval_run(ensemble,corr_level,init,n_members,result_list)

#make df out of list of lists 
df = pd.DataFrame(result_list,columns=['Corruption level','Initialisation','n_members',
    'acc','ece','nll','brier'])

save_path = os.path.join('logs','ensemble_results','resultsrtz.pkl')
df.to_pickle(save_path)


