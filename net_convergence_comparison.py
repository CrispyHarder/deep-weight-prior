# File to filter out runs using e.g. specific initialisations and comparing the 
# convergence and performance of all trained models  

import os
from turtle import title 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import yaml 
from datetime import date
import seaborn as sns 
import json

DATASET = 'pcam'

# given an array of files (who most likely have the same init for the networks) 
# it returns the avg accuracy and nll recorded on the test set
def print_test_perf(files):
    acc = []
    nll = []
    len_str = len(f'train-net-{DATASET}.py-logs.csv')
    rel_runs = [file[:-(len_str+1)] for file in files]
    for run in rel_runs:
        pdp = os.path.join(run,'perf_dict.json')
        with open(pdp,'r') as fp:
            dict = json.load(fp)
        acc.append(dict['acc'])
        nll.append(dict['nll'])
    acc = np.array(acc)
    nll = np.array(nll)
    acc_mean, acc_std = np.mean(acc), np.std(acc)
    nll_mean, nll_std = np.mean(nll), np.std(nll)
    # print(f'{init} has acc {acc_mean} pm {acc_std} and a nll of {nll_mean} pm {nll_std}' )
    return acc,nll 

#unused
def first_to_reach(data):
    global DATASET
    th = 0.80 if DATASET=='pcam' else 0.65
    for i in range(len(data)):
        if data[i]>=th:
            return i
    return 'dnf'

# gets the list of runs that were made for a dataset
logs_path = os.path.join('logs',f'exman-train-net-{DATASET}.py','runs')
runs = [os.path.join(logs_path,run) for run in os.listdir(logs_path)]

# The metrices that shall be considered 
# possible choices:
# 'NLL Train','ACC Train'
METR_NAMES = ['ACC Val']

#The initialisations that shall be considered-
#possible choices: 
#[['xavier'],['vae'],['he'],['vqvae1.0'],['vqvae1.3'],['vqvae1.0','pixelcnn0'],['vqvae1.3','pixelcnn0']]
#[['ghn_ce'],['ghn_loss']] # [['xavier'],['he'],['lvae'],['vae'],['vqvae1'],['vqvae1','pixelcnn'],['tvae'],['ghn_base'],['ghn_ce'],['ghn_loss']]
INIT_NAMES = [['xavier'],['he'],['vae'],['vqvae1','pixelcnn'],['ghn_base']]
# INIT_NAMES = [['ghn_base'],['ghn_ce']]
SAVE_PATH = os.path.join('logs','small-results','init comparison',str(date.today())) #the path where results are saved 
SAVE_SPEC = DATASET
SAVE_PLOTS = True
SHOW_PLOTS = True
STARTING_AT = 0 #where to start plotting if the training trajectories are of interest 
ENDING_AT = 20 #  121 if DATASET == 'cifar' else 41 #where to end when the training trajectories are of interest 

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

data = []
test_perf_acc = []
test_perf_nll = []

for i,init in enumerate(INIT_NAMES):
    files = []
    
    #filter out the runs that match the initialisation description 
    # and add the corresponding log file into the "files" list
    for run in runs:
        file  = os.path.join(run,f'train-net-{DATASET}.py-logs.csv')
        yaml_p = os.path.join(run,'params.yaml')
        with open(yaml_p) as f:
            dict = yaml.full_load(f)
        if not 'mult_init_prior' in dict:
            if dict['mult_init_mode'] == init[0]:
                files.append(file)
        elif 'mult_init_prior' in dict and len(init) ==1:
            if dict['mult_init_mode'] == init[0] and dict['mult_init_prior'] == '':
                files.append(file)
        elif 'mult_init_prior' in dict and len(init) ==2:
            if dict['mult_init_mode'] == init[0] and dict['mult_init_prior'] == init[1]:
                files.append(file)
    
    #get avg performance of best val models on test data
    acc,nll = print_test_perf(files)
    test_perf_acc.append(acc)
    test_perf_nll.append(nll)

    # read out the data from the log files 1nnltrain 2 acctrain 3nllval 4accval
    # the values are just added onto each other and then averaged at the end 
    init_data = np.genfromtxt(files[0],delimiter=',')[1:,1:5]
    for file in files[1:]:
        init_data += np.genfromtxt(file,delimiter=',')[1:,1:5]
    init_data = init_data / len(files)
    data.append(init_data)

# visual stuff
sns.set_theme()
sns.set_context('paper')
sns.set(rc={'figure.figsize':(10,8)})
sns.set(font_scale = 2.5)


#### The metrices (mostly ACCURACY) are getting PLOTTED below
# corresponding clear names and legend locations 
legends = {'NLL Train':'upper right','NLL Val':'upper right','ACC Val':'lower right','ACC Train':'lower right'}
labels = {'vae':'CVAE','he':'He','xavier':'Xavier','vqvae1':'VQVAE','vqvae1 + pixelcnn':'VQVAE*','tvae':'TVAE',
            'lvae':'LVAE','ghn_base':'GHN','ghn_loss':'Noise GHN[L] ','ghn_ce':'Noise GHN[CE]'}
COLUMN = 3 # validation accuracy
m_name = 'ACC Val'
ylabel = 'Accuracy'

# The plotting 
plt.figure()
for j,init in enumerate(INIT_NAMES):
    print(init,first_to_reach(data[j][:,3]))
    label = init[0] if len(init) == 1 else init[0] + ' + ' + init[1]
    label = labels[label]
    # lp = sns.lineplot(x=np.arange(STARTING_AT,ENDING_AT,1),y=data[j][STARTING_AT:ENDING_AT,3],label=label,lw=4,palette=['#8172b3','#937860'])
    lp = sns.lineplot(x=np.arange(STARTING_AT,ENDING_AT,1),y=data[j][STARTING_AT:ENDING_AT,3],label=label,lw=4)
lp.set(xlabel='Evaluation step')
lp.set(ylabel=ylabel)
lp.set(title='PCam')
plt.legend(loc=legends[m_name],fontsize=20)
plt.tight_layout()
if SAVE_PLOTS:    
    plt.savefig(os.path.join(SAVE_PATH,SAVE_SPEC+'_'+m_name+'.pdf'))
if SHOW_PLOTS:
    plt.show()


# Here the boxplots for the accuracies on the test set are made
labels_boxes = {'xavier':'Xavier','he':'He','vae':'VAE','vqvae1 + pixelcnn':"VQVAE*",'ghn_base':'GHN'}
#labels_boxes = {'ghn_ce':'Noise GHN[CE]','ghn_loss':'Noise GHN[L]'}

# test_perf_acc = pd.DataFrame(np.transpose(test_perf_acc))
# test_perf_acc.columns = labels_boxes.values()

# sns.set(rc={'figure.figsize':(10,8)})
# sns.set(font_scale = 2.5)

# plt.figure()
# b2 = sns.boxplot(data=test_perf_acc)
# b2.set(xlabel='Initialisation')
# b2.set(ylabel='Accuracy')
# b2.set(title='PatchCamelyon')
# plt.tight_layout()
# plt.savefig(os.path.join(SAVE_PATH,SAVE_SPEC+'_accuracy_boxplot.pdf'))
# plt.show()


## Unused nll stuff 

# test_perf_nll = pd.DataFrame(np.transpose(test_perf_nll))
# test_perf_nll.columns = labels_boxes.values()

# plt.figure()
# b1 = sns.boxplot(data=test_perf_nll)
# b1.set(xlabel='Initialisation')
# plt.title('NLL')
# plt.tight_layout()
# plt.savefig(os.path.join(SAVE_PATH,SAVE_SPEC+'_nll_boxplot.pdf'))
