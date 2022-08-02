import argparse
import torch
import os 
from utils import load_pcam_dataloaders
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

def eval_run(ensemble,init,result_list):
    '''gets a dataloader and an ensemble with n members 
    and produces 20 samples of that constellation, that are added 
    to a given list'''
    # get predictions and labels 

    labels_boxes = {'vae':'CVAE','he':'He','xavier':'Xavier','vqvae1':'VQVAE','vqvae1 + pixelcnn':"VQVAE\n+Pixel",'tvae':'TVAE',
            'lvae':'LVAE','ghn_base':'GHN','ghn_loss':'Noise\nGHN_1','ghn_ce':'Noise\nGHN_0'}

    init_name = init[0] if len(init) == 1 else init[0] + ' + ' + init[1]
    init_name = labels_boxes[init_name]
    for i in range(20):
        with torch.no_grad():
            logits, probs, labels = ensemble.get_ens_prediction(5,0)
        #get array of actualy predictions from the logits
        predictions = np.argmax(probs,axis=1)
        # get the confidences of each prediction = percentage of prediction
        confidences = [probs[i,int(predictions[i])] for i in range(len(logits))] 

        accuracy = np.sum(labels == predictions)/len(labels)
        ece = expected_calibration_error(confidences,predictions,labels)
        nll = float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)))
        brier = float(brier_multi(labels,probs))

        result_list.append([init_name,accuracy,ece,nll,brier])


#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id',type=int,choices=[0,1,2,3,4,5,6,7])
args = parser.parse_args()

#set device
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_,_,testloader = load_pcam_dataloaders(64)
dataloaders = [testloader]
result_list = [] 
for init in INIT_NAMES:
    ensemble = Ensemble(init,device,'pcam',dataloaders)
    eval_run(ensemble,init,result_list)

#make df out of list of lists 
df = pd.DataFrame(result_list,columns=['Initialisation','acc','ece','nll','brier'])

save_path = os.path.join('logs','ensemble_results','results_pcam.pkl')
df.to_pickle(save_path)


