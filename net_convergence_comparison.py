import os 
import pandas
import numpy as np
import matplotlib.pyplot as plt 
import yaml 

logs_path = os.path.join('logs','exman-train-net.py','runs')
runs = [os.path.join(logs_path,run) for run in os.listdir(logs_path) if run[:6] not in ['000001','000002']]
x_files = []
v_files = []
for run in runs:
    file  = os.path.join(run,'train-net.py-logs.csv')
    yaml_p = os.path.join(run,'params.yaml')
    with open(yaml_p) as f:
        dict = yaml.load(f)
    if dict['mult_init_mode'] == 'xavier':
        x_files.append(file)
    if dict['mult_init_mode'] == 'vae':
        v_files.append(file)

x_data = np.genfromtxt(x_files[0],delimiter=',')[1:,1:6]
for file in x_files[1:]:
    x_data += np.genfromtxt(file,delimiter=',')[1:,1:6]
x_data = x_data / len(x_files)

v_data = np.genfromtxt(v_files[0],delimiter=',')[1:,1:6]
for file in v_files[1:]:
    v_data += np.genfromtxt(file,delimiter=',')[1:,1:6]
v_data = v_data / len(v_files)
print(np.shape(v_data))
metr_names = ['loss','train_nll','test_nll','train_acc','test_acc']
inits = ['xavier','vae']
    
for i,m_name in enumerate(metr_names):
    plt.plot(x_data[:,i],label='xavier')
    plt.plot(v_data[:,i],label='vae')
    plt.title('{} convergence comparison'.format(m_name))
    plt.legend(loc='lower right')
    plt.savefig(os.path.join('..','..','small-results','16.9.2021','comparison convergence inits',m_name))
    plt.show()