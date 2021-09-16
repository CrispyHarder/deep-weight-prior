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

root_dir = os.path.join('..','..','small-results','16.9.2021','vae insights')
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
    root = os.path.join(root_dir,layer)
    os.makedirs(root,exist_ok=True)

    testloader, D = utils.get_dataloader(os.path.join(data_dir, 'test.npy'), test_bs, shuffle=False)
    prior = dist.Normal(torch.FloatTensor([0.]).to(vae.device), torch.FloatTensor([1.]).to(vae.device))

    #random samples from Xavier
    r_samples = []
    for i in range(25):
        w = torch.empty(3,3)
        nn.init.xavier_normal_(w)
        r_samples.append(w)
    ws = torch.stack(r_samples)
    ws = torch.reshape(ws,(25,1,3,3))
    f, axs = plt.subplots(nrows=5, ncols=5, figsize=(12, 10))
    for x, ax in zip(ws.reshape((-1, D, D)), axs.flat):
        sns.heatmap(tonp(x), ax=ax, cmap="bwr",vmin=-0.75,vmax=0.75)
        ax.axis('off')
    f.savefig(os.path.join(root, 'xavier samples'), dpi=200)
    plt.close(f)

    #samples 
    z = prior.rsample(sample_shape=(25, z_dim, 1))
    x_mu, x_var = vae.decode(z)
    f, axs = plt.subplots(nrows=5, ncols=5, figsize=(12, 10))
    samples = dist.Normal(x_mu, torch.sqrt(x_var)).rsample()
    for x, ax in zip(samples.reshape((-1, D, D)), axs.flat):
        sns.heatmap(tonp(x), ax=ax, cmap="bwr",vmin=-0.75,vmax=0.75)
        ax.axis('off')
    f.savefig(os.path.join(root, 'samples'), dpi=200)
    plt.close(f)

    # mean_reconstructions
    data = next(iter(testloader))
    data = data[:25].to(vae.device)
    [z_mu, z_var], [x_mu, x_var] = vae(data)
    f, axs = plt.subplots(nrows=5, ncols=5, figsize=(15, 7))
    for x, x_rec, ax in zip(data.reshape((-1, D, D)), x_mu.reshape((-1, D, D)), axs.flat):
        sns.heatmap(np.concatenate((tonp(x), tonp(x_rec)), 1), ax=ax, cmap="bwr",vmin=-0.75,vmax=0.75)
        ax.axis('off')
    f.savefig(os.path.join(root, 'mean_reconstructions'), dpi=200)
    plt.close(f)

    #sample_reconstructios 
    [z_mu, z_var], [x_mu, x_var] = vae(data)
    f, axs = plt.subplots(nrows=5, ncols=5, figsize=(15, 7))
    samples = dist.Normal(x_mu, torch.sqrt(x_var)).rsample()
    for x, x_rec, ax in zip(data.reshape((-1, D, D)), samples.reshape((-1, D, D)), axs.flat):
        sns.heatmap(np.concatenate((tonp(x), tonp(x_rec)), 1), ax=ax, cmap="bwr",vmin=-0.75,vmax=0.75)
        ax.axis('off')
    f.savefig(os.path.join(root, 'sample_reconstructions'), dpi=200)
    plt.close(f)