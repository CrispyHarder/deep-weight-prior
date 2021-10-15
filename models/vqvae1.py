import torch
import torch.nn as nn
from vq import VectorQuantizer,VectorQuantizerEMA


class Encoder3x3(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Encoder3x3, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(hidden_dim, z_dim, 3, padding=1),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder3x3(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Decoder3x3, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim, hidden_dim , 3, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(hidden_dim , hidden_dim, 3, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(hidden_dim, 1, 3, padding=1)
        )

    def forward(self, input):
        return self.decoder(input)


class VQVAE(nn.Module):
    def __init__(self, encoder, decoder, num_embeddings, commitment_cost, 
                    device=None, use_cuda=True, decay=0):
        super(VQVAE, self).__init__()
        # depricated
        self.use_cuda = use_cuda

        self.encoder = encoder
        self.decoder = decoder
        # we construct the vqvae in such a way that the z dim of the encoder is also the embedding dim
        self.embedding_dim = self.encoder.z_dim
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)
        self.device = device
        self.to(device)

    def encode(self, input):
        return self.encoder(input)

    def decode(self, input):
        return self.decoder(input)

    def forward(self, input):
        z = self.encoder(input)
        vq_loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self.decoder(quantized)

        return vq_loss, x_recon, perplexity
