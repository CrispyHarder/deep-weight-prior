import os
import torch
from models.tvae.grouper import Chi_Squared_from_Gaussian_2d
import torchvision

class TVAE(torch.nn.Module):
    def __init__(self, z_encoder, u_encoder, decoder, grouper):
        super(TVAE, self).__init__()
        self.z_encoder = z_encoder
        self.u_encoder = u_encoder
        self.decoder = decoder
        self.grouper = grouper
        
        self.device = grouper.device
        self.to(self.device)
        
    def forward(self, x):
        z, kl_z, _, _ = self.z_encoder(x)
        u, kl_u, _, _ = self.u_encoder(x)
        s = self.grouper(z, u)
        probs_x, neg_logpx_z = self.decoder(s, x)

        return z, u, s, probs_x, kl_z, kl_u, neg_logpx_z

    def get_IS_estimate(self, x, n_samples=100):
        log_likelihoods = []

        for n in range(n_samples):
            z, kl_z, log_q_z, log_p_z = self.z_encoder(x)
            u, kl_u, log_q_u, log_p_u = self.u_encoder(x)
            s = self.grouper(z, u)
            probs_x, neg_logpx_z = self.decoder(s, x)
            ll = (-1 * neg_logpx_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  + log_p_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  + log_p_u.flatten(start_dim=1).sum(-1, keepdim=True)
                  - log_q_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  - log_q_u.flatten(start_dim=1).sum(-1, keepdim=True))
            log_likelihoods.append(ll)
        ll = torch.cat(log_likelihoods, dim=-1)
        is_estimate = torch.logsumexp(ll, -1)
        return is_estimate

class VAE(TVAE):
    def get_IS_estimate(self, x, n_samples=100):
        log_likelihoods = []

        for n in range(n_samples):
            z, kl_z, log_q_z, log_p_z = self.z_encoder(x)
            s = self.grouper(z, torch.zeros_like(z))
            probs_x, neg_logpx_z = self.decoder(s, x)
            ll = (-1 * neg_logpx_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  + log_p_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  - log_q_z.flatten(start_dim=1).sum(-1, keepdim=True))
            log_likelihoods.append(ll)
        ll = torch.cat(log_likelihoods, dim=-1)
        is_estimate = torch.logsumexp(ll, -1)
        return is_estimate

    def forward(self, x):
        z, kl_z, _, _ = self.z_encoder(x)
        u = torch.zeros_like(z)
        kl_u = torch.zeros_like(kl_z)
        s = self.grouper(z, u)
        probs_x, neg_logpx_z = self.decoder(s, x)

        return z, u, s, probs_x, kl_z, kl_u, neg_logpx_z