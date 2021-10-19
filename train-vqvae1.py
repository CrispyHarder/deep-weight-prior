import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torch.serialization import save
import torch.nn.functional as F
import utils
from torch import nn
from utils import tonp
from logger import Logger
from models import vqvae1
import sys
import torch.distributions as dist
# import matplotlib.pyplot as plt
# import seaborn as sns
import myexman
from pathlib import Path
from my_utils import save_args_params

# sns.set()
# plt.switch_backend('agg')


def train(trainloader, testloader, vqvae, optimizer, scheduler, criterion, args, D):
    logger = Logger(name='logs', base=args.root)
    best_loss = 1000000
    for epoch in range(1, args.num_epochs + 1):
        adjust_learning_rate(optimizer, lr_linear(epoch))
        scheduler.step()
        train_recon_loss = utils.MovingMetric()
        train_vq_loss = utils.MovingMetric()
        train_perplexity = utils.MovingMetric()
        train_loss = utils.MovingMetric()

        for i, x in enumerate(trainloader):
            optimizer.zero_grad()
            x = x.to(vqvae.device)
            vq_loss, x_recon, perplexity = vqvae(x) 
            recon_loss = F.mse_loss(x,x_recon)
            loss = vq_loss + recon_loss

            loss.backward()
            optimizer.step()

            train_recon_loss.add(recon_loss.item(), x.size(0))
            train_vq_loss.add(vq_loss.item(), x.size(0))
            train_perplexity.add(perplexity.item(), x.size(0))
            train_loss.add(loss.item(),x.size())

        test_recon_loss = utils.MovingMetric()
        test_vq_loss = utils.MovingMetric()
        test_perplexity = utils.MovingMetric()
        test_loss = utils.MovingMetric()

        for i, x_test in enumerate(testloader):
            x_test = x_test.to(vqvae.device)
            test_vq_loss, test_x_recon, test_perplexity = vqvae(x_test)
            test_recon_loss = F.mse_loss(x_test,test_x_recon)
            test_loss = test_vq_loss + test_recon_loss

            test_recon_loss.add(test_recon_loss.item(), x_test.size(0))
            test_vq_loss.add(test_vq_loss.item(), x_test.size(0))
            test_perplexity.add(test_perplexity.item(), x_test.size(0))
            test_loss.add(test_loss.item(),x_test.size())

        test_likelihood = test_likelihood.get_val()
        
        train_recon_loss = train_recon_loss.get_val()
        test_recon_loss = test_recon_loss.get_val()
        train_vq_loss = train_vq_loss.get_val()
        test_vq_loss = test_vq_loss.get_val()
        train_perplexity = train_perplexity.get_val()
        test_perplexity = test_perplexity.get_val()
        train_loss = train_loss.get_val()
        test_loss = test_loss.get_val()

        logger.add_scalar(epoch, 'train_recon_loss', train_recon_loss)
        logger.add_scalar(epoch, 'train_vq_loss', train_vq_loss)
        logger.add_scalar(epoch, 'train_perplexity', train_perplexity)
        logger.add_scalar(epoch, 'train_loss', train_loss)

        logger.add_scalar(epoch, 'test_recon_loss', test_recon_loss)
        logger.add_scalar(epoch, 'test_vq_loss', test_vq_loss)
        logger.add_scalar(epoch, 'test_perplexity', test_perplexity)
        logger.add_scalar(epoch, 'test_loss', test_loss)

        logger.iter_info()
        logger.save()

        if (epoch-1) % 10 == 0:
            torch.save(vqvae.state_dict() , os.path.join(args.root, 'vqvae_params_epoch_{}.torch'.format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(args.root, 'opt_params_epoch_{}.torch'.format(epoch)))

        is_best = (test_loss < best_loss)
        if is_best:
            best_loss = test_loss
            torch.save(vqvae.state_dict(), os.path.join(args.root, 'vqvae_params.torch'))   
            if args.add_save_path : 
                torch.save(vqvae.state_dict(), os.path.join(args.add_save_path, 'vqvae_params.torch'))  
   

    torch.save(vqvae.state_dict(), os.path.join(args.root, 'vqvae_params_lastepoch.torch'))
    torch.save(optimizer.state_dict(), os.path.join(args.root, 'opt_params_lastepoch.torch'))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lr_linear(epoch):
    '''sets the learning rate in a way that it is lr at epoch 0 and linearily 
    decreases to 0 at args.num_epochs. After num_epochs the lr is constantly zero'''
    lr = args.lr * np.minimum((-epoch) * 1. / (args.num_epochs) + 1, 1.)
    return max(0, lr)


if __name__ == '__main__':
    parser = myexman.ExParser(file=__file__)
    parser.add_argument('--train')
    parser.add_argument('--test')
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--resume', default='')
    parser.add_argument('--resume_opt', default='')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--test_bs', default=512, type=int)
    parser.add_argument('--z_dim', default=8, type=int)
    parser.add_argument('--hidden_dim', default=16, type=int)
    parser.add_argument('--kernel_dim', default=16, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--lr_decay_step', default=int(11e8), type=int)
    parser.add_argument('--decay', default=0.5, type=float)
    parser.add_argument('--var', default='train')
    parser.add_argument('--add_save_path', default='')
    parser.add_argument('--num_embeddings', type=int, default=128)
    parser.add_argument('--ema_decay', type= float, default=0.)
    parser.add_argument('--commitment_cost', type=float, default=0.25)
    parser.add_argument('--vqvae_spec',type=str, default= 1 , 
            help = 'not needed for direct use')
    parser.add_argument('--vqvae_arch',type=str, default= 100 , 
            help = 'not needed for direct use')
    args = parser.parse_args()

    #save the args to the dict, from where the vaes are initialised
    save_args_params(args,args.add_save_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.data_dir:
        trainloader, D = utils.get_dataloader(os.path.join(args.data_dir, 'train.npy'), args.batch_size, shuffle=True)
        testloader, D = utils.get_dataloader(os.path.join(args.data_dir, 'test.npy'), args.test_bs, shuffle=False)
    else:
        trainloader, D = utils.get_dataloader(args.train, args.batch_size, shuffle=True)
        testloader, D = utils.get_dataloader(args.test, args.test_bs, shuffle=False)

    assert args.kernel_dim == D, '--kernel-dim != D (in dataset)'

    if D == 3:
        decoder = vqvae1.Decoder3x3(args.z_dim, args.hidden_dim)
        encoder = vqvae1.Encoder3x3(args.z_dim, args.hidden_dim)

    vqvae = vqvae1.VQVAE(encoder, decoder, args.num_embeddings, args.commitment_cost, 
            device=device, decay=args.ema_decay)
    if args.resume_vae:
        vqvae.load_state_dict(torch.load(args.resume_vae))

    optimizer = torch.optim.Adam(vqvae.parameters(), lr=args.lr)
    if args.resume_opt:
        optimizer.load_state_dict(torch.load(args.resume_opt))
        optimizer.param_groups[0]['lr'] = args.lr

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.decay)
    criterion = utils.VAEELBOLoss(use_cuda=args.cuda)

    train(trainloader, testloader, vqvae, optimizer, scheduler, criterion, args, D)
