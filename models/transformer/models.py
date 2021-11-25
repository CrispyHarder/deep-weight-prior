''' Define the Transformer model '''
import torch
from torch._C import dtype
import torch.nn as nn
import numpy as np
from models.transformer.layers import EncoderLayer, DecoderLayer
from models.transformer.vae_layer import VAE_layer
from models.vae import VAE

__author__ = "Yu-Hsiang Huang"


def get_pad_mask(seq):
    '''in order to pad the padding_idx in the language sequences'''
    return (seq != 0.).unsqueeze(-2)


def get_subsequent_mask(seq,length):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_embeddings, embedding_dim, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_position=9):

        super().__init__()

        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.src_emb = nn.Linear(n_position,n_embeddings*embedding_dim,bias=False)
        self.position_enc = PositionalEncoding(self.embedding_dim, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_emb(src_seq).view(-1,self.n_embeddings,self.embedding_dim)
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_embeddings, embedding_dim, n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_position=9, dropout=0.1):

        super().__init__()

        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.trg_emb = nn.Conv2d(1,self.n_embeddings*self.embedding_dim,3)
        self.position_enc = PositionalEncoding(self.embedding_dim, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_emb(trg_seq).view(-1,self.n_embeddings,self.embedding_dim)
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_embeddings, embedding_dim, 
            d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            device=None):

        super().__init__()

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        self.d_model = d_model
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.device = device

        self.encoder = Encoder(            
            n_embeddings=n_embeddings, embedding_dim=embedding_dim, 
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            d_model=d_model, d_inner=d_inner, dropout=dropout, n_position=n_position)

        self.vae_layer = VAE_layer(self.d_model,self.device)

        self.decoder = Decoder(
            n_embeddings=n_embeddings, embedding_dim=embedding_dim, 
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            d_model=d_model, d_inner=d_inner, dropout=dropout, n_position=n_position)

        self.trg_prj = nn.ConvTranspose2d(self.embedding_dim*self.n_embeddings,1,3)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        self.to(device)

    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq,self.n_embeddings)
        trg_mask = get_pad_mask(trg_seq,self.n_embeddings) & get_subsequent_mask(trg_seq,self.n_embeddings)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        z, kl = self.vae_layer(enc_output)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, z, src_mask)
        dec_output = dec_output.view(-1,self.n_embeddings*self.embedding_dim,1,1)
        seq_recon = self.trg_prj(dec_output)

        return seq_recon, kl  #seq_recon.view(-1, seq_recon.size(2))

    def generate(self,batch_size):
        '''samples from prior, reshapes and sends samples through decoder'''
        pr_samples = torch.randn((batch_size, self.n_embeddings, self.embedding_dim)).to(self.device)
        for i in range(self.device):
            pass