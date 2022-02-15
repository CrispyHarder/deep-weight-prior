import os
import numpy as np
import torch
import torch.nn.functional as F
import utils
from logger import Logger
import myexman
from my_utils import save_args_params
from models import vae
from torch.utils.tensorboard import SummaryWriter
import torch.distributions as dist

def train(trainloader, testloader, vae, optimizer, args, writer):
    logger = Logger(name='logs', base=args.root)
    best_elbo = -1000000
    prior = dist.Normal(torch.FloatTensor([0.]).to(vae.device), torch.FloatTensor([1.]).to(vae.device))

    test_kl = utils.MovingMetric()
    test_likelihood = utils.MovingMetric()
    test_loss = utils.MovingMetric()

    for i, x in enumerate(testloader):
        x = x.to(vae.device)
        [z_mu, z_var], [x_mu, x_var] = vae(x)
        
        test_likelihood.add(dist.Normal(x_mu, torch.sqrt(x_var)).log_prob(x).sum().item(), x.size(0))
        test_kl.add(dist.kl_divergence(dist.Normal(z_mu, torch.sqrt(z_var)), prior).sum().item(), x.size(0))

    test_kl = test_kl.get_val()
    test_likelihood = test_likelihood.get_val()
    test_loss = test_kl - test_likelihood

    logger.add_scalar(0, 'test_kl', test_kl)
    logger.add_scalar(0, 'test_likelilhod', test_likelihood)
    logger.add_scalar(0, 'test_loss', test_loss)
    logger.iter_info()
    logger.save()

    writer.add_scalar('test/kl',test_kl,0)
    writer.add_scalar('test/likelihod',test_likelihood,0)
    writer.add_scalar('test/loss',test_loss,0)

    torch.save(vae.state_dict(), os.path.join(args.root, 'vae_params_init.torch'))

    for epoch in range(1, args.num_epochs + 1):

        #set learning rate
        adjust_learning_rate(optimizer, lr_linear(epoch-1))

        train_likelihood = utils.MovingMetric()
        train_kl = utils.MovingMetric()
        train_loss = utils.MovingMetric()

        for i, x in enumerate(trainloader):
            optimizer.zero_grad()
            x = x.to(vae.device)
            [z_mu, z_var], [x_mu, x_var] = vae(x)
            
            likelihood = dist.Normal(x_mu, torch.sqrt(x_var)).log_prob(x).sum()
            kl = dist.kl_divergence(dist.Normal(z_mu, torch.sqrt(z_var)), prior).sum()
            loss = -likelihood + kl

            loss.backward()
            optimizer.step()

            train_likelihood.add(likelihood.item(), x.size(0))
            train_kl.add(kl.item(), x.size(0))
            train_loss.add(loss.item(), x.size(0))

        test_kl = utils.MovingMetric()
        test_likelihood = utils.MovingMetric()
        test_loss = utils.MovingMetric()

        for i, x in enumerate(testloader):
            x = x.to(vae.device)
            [z_mu, z_var], [x_mu, x_var] = vae(x)
           
            test_likelihood.add(dist.Normal(x_mu, torch.sqrt(x_var)).log_prob(x).sum().item(), x.size(0))
            test_kl.add(dist.kl_divergence(dist.Normal(z_mu, torch.sqrt(z_var)), prior).sum().item(), x.size(0))
        
        train_kl = train_kl.get_val()
        train_likelihood = train_likelihood.get_val()
        train_loss = train_loss.get_val()
        test_kl = test_kl.get_val()
        test_likelihood = test_likelihood.get_val()
        test_loss = test_kl - test_likelihood

        logger.add_scalar(epoch, 'train_kl', train_kl)
        logger.add_scalar(epoch, 'train_likelyhood', train_likelihood)
        logger.add_scalar(epoch, 'train_loss', train_loss)
        logger.add_scalar(epoch, 'test_kl', test_kl)
        logger.add_scalar(epoch, 'test_likelilhod', test_likelihood)
        logger.add_scalar(epoch, 'test_loss', test_loss)
        logger.iter_info()
        logger.save()

        writer.add_scalar('train/kl',train_kl,epoch)
        writer.add_scalar('train/likelihood',train_likelihood,epoch)
        writer.add_scalar('train/loss',train_loss,epoch)
        writer.add_scalar('test/kl',test_kl,epoch)
        writer.add_scalar('test/likelihod',test_likelihood,epoch)
        writer.add_scalar('test/loss',test_loss,epoch)

        test_elbo = test_likelihood - test_kl
        is_best = (test_elbo>best_elbo)
        if is_best:
            best_elbo = test_elbo
            torch.save(vae.state_dict(), os.path.join(args.root, 'vae_params.torch'))   
            if args.add_save_path : 
                torch.save(vae.state_dict(), os.path.join(args.add_save_path, 'vae_params.torch'))  

    torch.save(vae.state_dict(), os.path.join(args.root, 'vae_params_lastepoch.torch'))
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
    #train settings 
    parser.add_argument('--data_dir', default='')

    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu_id', default='0')

    #optimisation
    parser.add_argument('--weight_decay', type=float, default=1.)
    parser.add_argument('--lr', default=0.01, type=float)
    
    #evaluation
    parser.add_argument('--test_bs', default=512, type=int)

    #model specifics
    parser.add_argument('--z_dim', default=8, type=int)
    parser.add_argument('--hidden_dim', default=32, type=int)
    
    #misc
    parser.add_argument('--add_save_path', default='')

    args = parser.parse_args()

    #save the args to the dict, from where the vaes are initialised
    save_args_params(args,args.add_save_path)

    #et GPU, device and seeds
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #get dataloaders
    trainloader, D = utils.get_dataloader(os.path.join(args.data_dir, 'train.npy'), args.batch_size, shuffle=True)
    testloader, D = utils.get_dataloader(os.path.join(args.data_dir, 'test.npy'), args.test_bs, shuffle=False)

    decoder = vae.Decoder3x3(args.z_dim, args.hidden_dim)
    encoder = vae.Encoder3x3(args.z_dim, args.hidden_dim)
    vae = vae.VAE(encoder, decoder, device=device)
    
    #configure optimisation
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = utils.VAEELBOLoss(use_cuda=args.cuda) 

    writer = SummaryWriter(args.root)

    train(trainloader, testloader, vae, optimizer, args, writer)

    writer.flush()
