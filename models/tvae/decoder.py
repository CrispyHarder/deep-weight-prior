import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal

class Decoder(nn.Module):
    def __init__(self, model):
        super(Decoder, self).__init__()
        self.model = model

    def forward(self, x):
        raise NotImplementedError


class Bernoulli_Decoder(Decoder):
    def __init__(self, model):
        super(Bernoulli_Decoder, self).__init__(model)

    def forward(self, z, x):
        probs_x = torch.clamp(self.model(z), 0, 1)
        p = Bernoulli(probs=probs_x)
        neg_logpx_z = -1 * p.log_prob(x)

        return probs_x, neg_logpx_z

    def only_decode(self, z):
        probs_x = torch.clamp(self.model(z), 0, 1)
        return probs_x

class Gaussian_Decoder(Decoder):
    def __init__(self, model, scale=1.0):
        super(Gaussian_Decoder, self).__init__(model)
        self.scale = torch.tensor([scale])

    def forward(self, z, x):
        x_recon = self.model(z)
        recon_loss = F.mse_loss(x,x_recon)

        return x_recon, recon_loss

    def only_decode(self, z):
        mu_x = self.model(z)
        return mu_x