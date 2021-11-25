from typing import ForwardRef
import torch
import torch.nn as nn 
import torch.distributions as dist
import torch.nn.functional as F
from torch.nn import init

from models.vae import DecoderId

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)

class LVAE(nn.Module):
    def __init__(self,dim_list, z_dim, device) -> None:
        super().__init__()
        self.dim_list = [9] + dim_list
        self.dim_list_rev = [z_dim] + self.dim_list[::-1]
        self.z_dim = z_dim 
        self.device = device

        self.prior = dist.Normal(torch.FloatTensor([0.]).to(device), torch.FloatTensor([1.]).to(device))

        self.encoder = []
        for i in range(len(self.dim_list)-1):
            self.encoder.append(nn.Linear(self.dim_list[i],self.dim_list[i+1]))
            self.encoder.append(nn.ReLU()) 
        self.encoder = nn.Sequential(*self.encoder)

        self.mu = nn.Linear(self.dim_list[-1],z_dim)
        self.var = nn.Linear(self.dim_list[-1],z_dim)

        self.decoder = []
        for i in range(len(self.dim_list_rev)-1):
            self.decoder.append(nn.Linear(self.dim_list_rev[i],self.dim_list_rev[i+1]))
            self.decoder.append(nn.ReLU())
        self.decoder = nn.Sequential(*self.decoder)

        self.apply(_weights_init)
        self.to(device)

    def forward(self,x):
        hidden = self.encoder(x)
        mu = self.mu(hidden)
        var = F.softplus(self.var(hidden))

        z_dist = dist.Normal(mu, torch.sqrt(var))

        z = z_dist.rsample().to(self.device)
        kl_loss = dist.kl_divergence(z_dist, self.prior).sum()

        x_recon = self.decoder(z)
        recon_loss = F.mse_loss(x,x_recon)
        return x_recon, recon_loss, kl_loss
        

