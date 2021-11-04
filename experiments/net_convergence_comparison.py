# File to filter out runs using e.g. specific initialisations and comparing the 
# convergence of multiple models 

import os 
import pandas
import numpy as np
import matplotlib.pyplot as plt 
import yaml 
from datetime import date

logs_path = os.path.join('logs','exman-train-net.py','runs')
runs = [os.path.join(logs_path,run) for run in os.listdir(logs_path) if run[:6] not in ['000001','000002']]

#[('xavier'),('vae'),('he'),('vqvae1.0'),('vqvae1.0','pixelcnn0'),('vqvae1.2')]
INIT_NAMES = [('xavier'),('vae'),('he'),('vqvae1.0'),('vqvae1.0','pixelcnn0')] 
METR_NAMES = ['loss','train_nll','test_nll','train_acc','test_acc']
SAVE_PATH = os.path.join('..','..','small-results',str(date.today()),'init comparison')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

data = []

for i,init in enumerate(INIT_NAMES):
    files = []
    
    for run in runs:
        file  = os.path.join(run,'train-net.py-logs.csv')
        yaml_p = os.path.join(run,'params.yaml')
        with open(yaml_p) as f:
            dict = yaml.load(f)
        if len(init) == 1:
            if dict['mult_init_mode'] == init[0]:
                files.append(file)
        elif len(init) == 2:
            if dict['mult_init_mode'] == init[0] and dict['mult_init_prior'] == init[1]:
                files.append(file)

        
    init_data = np.genfromtxt(files[0],delimiter=',')[1:,1:6]
    for file in files[1:]:
        init_data += np.genfromtxt(file,delimiter=',')[1:,1:6]
    init_data = init_data / len(files)
    data.append(init_data)

for i,m_name in enumerate(METR_NAMES):
    for j,init in enumerate(INIT_NAMES):
        label = init[0] + ' + ' + init[1]
        plt.plot(data[j][:,i],label=label)
    plt.title('{} convergence comparison'.format(m_name))
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(SAVE_PATH,m_name))
    plt.show()