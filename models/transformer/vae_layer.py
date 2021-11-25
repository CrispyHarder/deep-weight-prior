import torch
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F


class VAE_layer(nn.Module):
    '''simple layer to make latent space operations '''
    def __init__(self, d_model, device):
        super(VAE_layer, self).__init__()
        self.d_model = d_model

        self.fc_mu = nn.Linear(d_model,d_model)
        self.fc_var = nn.Linear(d_model,d_model)

        self.prior = dist.Normal(torch.FloatTensor([0.]).to(device), torch.FloatTensor([1.]).to(device))

    def forward(self, enc_output):
        mu = self.fc_mu(enc_output)
        var = F.softplus(self.fc_var(enc_output))

        z = dist.Normal(mu, torch.sqrt(var)).rsample()

        kl = dist.kl_divergence(dist.Normal(mu, torch.sqrt(var)), self.prior).sum()
        return z, kl