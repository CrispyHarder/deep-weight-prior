import os 
import torch 
import argparse 
import yaml
import utils 
import matplotlib.pyplot as plt 
import seaborn as sns 
import math
import numpy as np
from datetime import date 

def plot_matrix_as_heatmap(matrix,show=False,title='',xlabel='',ylabel='',save_path=''):
    '''plots the cosine similariy matrix of a number of models
    or model configurations'''
    n = np.shape(np.array(matrix))[0]
    ticks = math.floor(n/4)
    sns.set_theme()
    ax = sns.heatmap(matrix,xticklabels=ticks,yticklabels=ticks,cmap='bwr')
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

logs_path = os.path.join('logs','exman-train-net.py','runs')
runs = [os.path.join(logs_path,run) for run in os.listdir(logs_path) if run[:6] not in ['000001','000002']]

INIT_NAMES = [['vae'],['ghn_default']]
SAVE_PATH = os.path.join('..','..','small-results',str(date.today()),'prediction_similarity')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

parser = argparse.ArgumentParser()
parser.add_argument('--init',type=str)
parser.add_argument('--device')
parser.add_argument('--sim',choices=['pred','logits'])
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.manual_seed_all(42)
torch.manual_seed(42)

init = args.init

model_paths = []
for run in runs:
    file  = os.path.join(run,'net_params.torch')
    yaml_p = os.path.join(run,'params.yaml')
    with open(yaml_p) as f:
        dict = yaml.full_load(f)
    if not 'mult_init_prior' in dict:
        if dict['mult_init_mode'] == init:
            model_paths.append(file)

_, testloader = utils.load_dataset(data='cifar', train_bs=64, test_bs=500, num_examples=None, seed=42,augmentation=False)

if args.sim == 'pred':
    all_predictions = []
    for model_path in model_paths:
        if init == 'vae':
            model = utils.load_vae(model_path,device)

        predictions = []
        for x,_ in testloader:
            x = x.to(device)
            p = model(x)
            predictions.append(p.max(1)[1])
        predictions = torch.cat(predictions)
        all_predictions.append(predictions)
    all_predictions = torch.stack(all_predictions)

    length_data = all_predictions.shape[1]
    matrix = torch.zeros(length_data,length_data)
    for i in range(length_data):
        for j in range(i+1):
            pred_sim = torch.sum(all_predictions[i] == all_predictions[j])/length_data
            matrix[i,j] = matrix[j,i] = pred_sim

if args.sim == 'logits':
    CosineSimilarity = torch.nn.CosineSimilarity(dim=0)
    all_predictions = []
    for model_path in model_paths:
        if init == 'vae':
            model = utils.load_vae(model_path,device)

        predictions = []
        for x,_ in testloader:
            x = x.to(device)
            p = model(x)
            predictions.append(torch.flatten(p))
        predictions = torch.cat(predictions)
        all_predictions.append(predictions)
    all_predictions = torch.stack(all_predictions)

    length_data = all_predictions.shape[1]
    matrix = torch.zeros(length_data,length_data)
    for i in range(length_data):
        for j in range(i+1):
            cos_sim = CosineSimilarity(all_predictions[i],all_predictions[j])
            matrix[i,j] = matrix[j,i] = cos_sim

title = f'{args.sim} Similarity of {args.init} inits'
save_path = os.path.join(SAVE_PATH,title)
plot_matrix_as_heatmap(matrix,title=title,save_path=save_path)