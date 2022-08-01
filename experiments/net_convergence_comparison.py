# File to filter out runs using e.g. specific initialisations and comparing the 
# convergence of multiple models 

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import yaml 
from datetime import date
import seaborn as sns 
import json

DATASET = 'pcam'

def print_test_perf(files,init):
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
    


logs_path = os.path.join('logs',f'exman-train-net-{DATASET}.py','runs')
runs = [os.path.join(logs_path,run) for run in os.listdir(logs_path)]

#[['xavier'],['vae'],['he'],['vqvae1.0'],['vqvae1.3'],['vqvae1.0','pixelcnn0'],['vqvae1.3','pixelcnn0']]
INIT_NAMES = [['vae'],['he'],['xavier'],['vqvae1'],['vqvae1','pixelcnn'],['tvae'],['lvae'],['ghn_base'],['ghn_loss'],['ghn_ce']]
METR_NAMES = ['ACC Val'] # 'NLL Train','ACC Train',
SAVE_PATH = os.path.join('logs','small-results',str(date.today()),'init comparison')
SAVE_SPEC = DATASET+'ff'
SAVE_PLOTS = True
SHOW_PLOTS = True
STARTING_AT = 0
ENDING_AT = 16 if DATASET == 'cifar' else 10

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

data = []
test_perf_acc = []
test_perf_nll = []

for i,init in enumerate(INIT_NAMES):
    files = []
    
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
    
    #print avg performance of best val models on test data
    acc,nll = print_test_perf(files,init)
    test_perf_acc.append(acc)
    test_perf_nll.append(nll)

    init_data = np.genfromtxt(files[0],delimiter=',')[1:,1:5]
    for file in files[1:]:
        init_data += np.genfromtxt(file,delimiter=',')[1:,1:5]
    init_data = init_data / len(files)
    data.append(init_data)

sns.set_theme()
sns.set_context('paper')
sns.set(rc={'figure.figsize':(10,8)})
sns.set(font_scale = 1.9)


titles ={'NLL Train':' ','NLL Val':'NLL','ACC Val':'Accuracy','ACC Train':' '}
legends = {'NLL Train':'upper right','NLL Val':'upper right','ACC Val':'lower right','ACC Train':'lower right'}
labels = {'vae':'CVAE','he':'He','xavier':'Xavier','vqvae1':'VQVAE','vqvae1 + pixelcnn':'VQVAE + Pixel','tvae':'TVAE',
            'lvae':'LVAE','ghn_base':'GHN','ghn_loss':'Noise GHN[L]','ghn_ce':'Noise GHN[C]'}
COLUMN = 3
for i,m_name in enumerate(METR_NAMES):
    plt.figure()
    for j,init in enumerate(INIT_NAMES):
        label = init[0] if len(init) == 1 else init[0] + ' + ' + init[1]
        label = labels[label]
        sns.lineplot(x=np.arange(STARTING_AT,ENDING_AT,1),y=data[j][STARTING_AT:ENDING_AT,COLUMN],label=label)
    plt.title(titles[m_name])
    plt.legend(loc=legends[m_name])
    plt.tight_layout()
    if SAVE_PLOTS:    
        plt.savefig(os.path.join(SAVE_PATH,SAVE_SPEC+'_'+m_name+'.pdf'))
    if SHOW_PLOTS:
        plt.show()


# labels_boxes = {'vae':'CVAE','he':'He','xavier':'Xavier','vqvae1':'VQVAE','vqvae1 + pixelcnn':"VQVAE\n+Pixel",'tvae':'TVAE',
#             'lvae':'LVAE','ghn_base':'GHN','ghn_loss':'Noise\nGHN[L]','ghn_ce':'Noise\nGHN[C]'}

# test_perf_acc = pd.DataFrame(np.transpose(test_perf_acc))
# test_perf_acc.columns = labels_boxes.values()

# test_perf_nll = pd.DataFrame(np.transpose(test_perf_nll))
# test_perf_nll.columns = labels_boxes.values()

# sns.set(rc={'figure.figsize':(11,8)})
# sns.set(font_scale = 1.6)

# plt.figure()
# b1 = sns.boxplot(data=test_perf_nll)
# b1.set(xlabel='Initialisation')
# plt.title('NLL')
# plt.tight_layout()
# plt.savefig(os.path.join(SAVE_PATH,SAVE_SPEC+'_nll_boxplot.pdf'))


# plt.figure()
# b2 = sns.boxplot(data=test_perf_acc)
# b2.set(xlabel='Initialisation')
# plt.title('Accuracy')
# plt.tight_layout()
# plt.savefig(os.path.join(SAVE_PATH,SAVE_SPEC+'_accuracy_boxplot.pdf'))
# plt.show()



