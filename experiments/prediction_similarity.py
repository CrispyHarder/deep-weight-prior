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
from models.cifar.resnet import resnet20
from models.cifar import ResNet

def plot_matrix_as_heatmap(matrix,show=False,title='',save_path='',cbar=False,label=False,sim_name=''):
    '''plots the cosine similariy matrix of a number of models
    or model configurations'''
    global VMIN 
    plt.figure()
    n = np.shape(np.array(matrix))[0]
    sns.set_theme()
    sns.set_context('paper')
    ticks = np.arange(VMIN+0.05,1.01,0.05)
    ax = sns.heatmap(matrix,cmap='coolwarm',square=True,cbar=cbar,vmin=VMIN, cbar_kws={'ticks':ticks}) 
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    ax.set_title(title,fontsize=FONTSZ)
    if label:
        ax.set_ylabel(sim_name,fontsize=25)
    if cbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=FONTSZ)
    
    #plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    sns.set_theme()
    sns.set_context('paper')
    # sns.set(rc={'figure.figsize':(10,8)})
    sns.set(font_scale = 30)

    if save_path:
        plt.savefig(save_path,bbox_inches = 'tight',
            pad_inches = 0)
    if show:
        plt.show()

def load_resnet(path,ds,device=None):
    n_classes = 10 if ds=='cifar' else 2
    model = ResNet([3,3,3],num_classes=n_classes).to(device)
    model.load_state_dict(torch.load(path,map_location=device))
    model = model.to(device)
    return model

def compute_avg(matrix):
    dim = matrix.shape[0]
    vals = []
    for i in range(dim):
        for j in np.arange(0,i,1):
            vals.append(matrix[i,j])
    vals = np.array(vals)
    return round(np.mean(vals),4),np.std(vals)

def save_matrix(matrix,ds,sim,init):
    global SAVE_PATH
    torch.save(matrix,os.path.join(SAVE_PATH,f'matrix_{ds}_{sim}_{init}.torch'))

def load_matrix(ds,sim,init):
    global SAVE_PATH
    matrix = torch.load(os.path.join(SAVE_PATH,f'matrix_{ds}_{sim}_{init}.torch'))
    return matrix 

parser = argparse.ArgumentParser()
parser.add_argument('--ds',choices=['pcam','cifar'],type=str)
parser.add_argument('--gpu_id')
parser.add_argument('--sim',choices=['pred','logits'])
parser.add_argument('--comp',action='store_true',default=False)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.manual_seed_all(42)
torch.manual_seed(42)

logs_path = os.path.join('logs',f'exman-train-net-{args.ds}.py','runs')
runs = [os.path.join(logs_path,run) for run in os.listdir(logs_path)]

if args.ds =='pcam':
    VMIN = 0.93 if args.sim=='logits' else 0.95
else : 
    VMIN = 0.7 if args.sim=='logits' else 0.8
FONTSZ = 30

INITS = [['he'],['vqvae1','pixelcnn'],['ghn_base'],['ghn_ce']]
INIT_NAMES = {'he':'He','tvae':'TVAE','ghn_base':'GHN','ghn_ce':'Noise GHN','vqvae1':'VQVAE*'}
EXPERIMENT_NAME = 'prediction similarity'
SAVE_PATH = os.path.join('logs','small-results',EXPERIMENT_NAME,str(date.today())) #SAVE_PATH = os.path.join('logs','small-results',EXPERIMENT_NAME,str(date.today()))
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)




if args.comp:
    if args.ds == 'pcam':
        _,_,testloader = utils.load_pcam_dataloaders(64)
    else:
        _,_,testloader = utils.load_cifar10_loaders(64,500)

    for init in INITS:

        model_paths = []
        for run in runs:
            file  = os.path.join(run,'net_params.torch')
            yaml_p = os.path.join(run,'params.yaml')
            with open(yaml_p) as f:
                dict = yaml.full_load(f)
            if not 'mult_init_prior' in dict:
                if dict['mult_init_mode'] == init[0]:
                    model_paths.append(file)
            elif 'mult_init_prior' in dict and len(init) ==1:
                if dict['mult_init_mode'] == init[0] and dict['mult_init_prior'] == '':
                    model_paths.append(file)
            elif 'mult_init_prior' in dict and len(init) ==2:
                if dict['mult_init_mode'] == init[0] and dict['mult_init_prior'] == init[1]:
                    model_paths.append(file)
            if dict['mult_init_mode'] == init:
                model_paths.append(file)



        # get matrix to make the heatmap 
        if args.sim == 'pred':
            all_predictions = []
            for model_path in model_paths: 
                model = load_resnet(model_path,args.ds,device)
                model.eval()
                predictions = []
                with torch.no_grad():
                    for i,(x,_) in enumerate(testloader):
                        x = x.to(device)
                        p = model(x)
                        predictions.append(p.max(1)[1])
                predictions = torch.cat(predictions)
                all_predictions.append(predictions)
            all_predictions = torch.stack(all_predictions)

            num_models = all_predictions.shape[0]
            length_data = all_predictions.shape[1]
            matrix = torch.zeros(num_models,num_models)
            for i in range(num_models):
                for j in range(i+1):
                    pred_sim = torch.sum(all_predictions[i] == all_predictions[j])/length_data
                    matrix[i,j] = matrix[j,i] = pred_sim

        if args.sim == 'logits':
            CosineSimilarity = torch.nn.CosineSimilarity(dim=0)
            all_predictions = []
            for model_path in model_paths: 
                model = load_resnet(model_path,args.ds,device)
                model.eval()
                predictions = []
                with torch.no_grad():
                    for i,(x,_) in enumerate(testloader):
                        x = x.to(device)
                        p = model(x)
                        predictions.append(torch.flatten(p))
                predictions = torch.cat(predictions)
                all_predictions.append(predictions)
            all_predictions = torch.stack(all_predictions)

            num_models = all_predictions.shape[0]
            matrix = torch.zeros(num_models,num_models)
            for i in range(num_models):
                for j in range(i+1):
                    cos_sim = CosineSimilarity(all_predictions[i],all_predictions[j])
                    matrix[i,j] = matrix[j,i] = cos_sim
        save_matrix(matrix,args.ds,args.sim,init)


for i,init in enumerate(INITS):
    #whether to draw a colormap
    cbar = True if i+1 == len(INITS) else False
    label = True if i == 0 else False
    #p_title = True if args.sim=='pred' else False
    sim_name = 'Prediction similarity' if args.sim =='pred' else 'Cosine similarity of logits'
    matrix = load_matrix(args.ds,args.sim,init)
    title = f'{args.sim} Similarity of {init[0]} inits'
    avg,std = compute_avg(matrix)
    full_title = INIT_NAMES[init[0]]+' ('+str(avg)+')'
    save_path = os.path.join(SAVE_PATH,args.ds+' '+title)
    plot_matrix_as_heatmap(matrix,title=full_title,save_path=save_path,cbar=cbar,label=label,sim_name=sim_name)
plt.show()