import os
import numpy as np
import torch
import utils
from utils import tonp
import torch.distributions as dist
import matplotlib.pyplot as plt
import seaborn as sns
import yaml 
import torch.nn as nn

root_dir = os.path.join('..','..','small-results','7.10.2021','recon vs mean of vae')
for run in os.listdir(os.path.join('logs','exman-train-vae.py','runs')):
    run_path = os.path.join('logs','exman-train-vae.py','runs',run)
    params_path = os.path.join(run_path,'params.yaml')
    with open(params_path) as file:
        params_dic = yaml.full_load(file)
    layer = params_dic['data_dir'].rsplit('/')[-2]
    vae = utils.load_vae(os.path.join(run_path),device=torch.device('cpu'))
    data_dir = os.path.join('data','resnet20','3x3','layer_{}'.format(layer[-1]),'conv')
    test_bs = 512
    z_dim = 2

    testloader, D = utils.get_dataloader(os.path.join(data_dir, 'test.npy'), test_bs, shuffle=False)
    prior = dist.Normal(torch.FloatTensor([0.]).to(vae.device), torch.FloatTensor([1.]).to(vae.device))
    tuples = []
    for i,data in enumerate(testloader):
        data = data[:25].to(vae.device)
        [z_mu, z_var], [x_mu, x_var] = vae(data)
        for x, x_rec in zip(data.reshape((-1, D, D)), x_mu.reshape((-1, D, D))):
            tuples.append([torch.linalg.norm(torch.flatten(x)),torch.linalg.norm(torch.flatten(x-x_rec))])
    tuples = np.array(tuples)
    plt.figure()
    plt.scatter(tuples[:,0],tuples[:,1])
    plt.xlim([0,3.5])
    plt.ylim([0,3.5])
    plt.title('mean of slice vs recon error in {}'.format(layer))
    plt.xlabel('mean of slice')
    plt.ylabel('reconstruction')
    plt.savefig(os.path.join(root_dir, layer), dpi=200)
            

