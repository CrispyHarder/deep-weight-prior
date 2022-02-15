import os
import numpy as np
import torch
import utils
from logger import Logger
import myexman
from my_utils import save_args_params
from models.tvae.encoder import Gaussian_Encoder
from models.tvae.decoder import Gaussian_Decoder
from models.tvae.models import MLP_Decoder,MLP_Encoder
from models.tvae.grouper import Stationary_Capsules_1d
from models.tvae.tvae import TVAE
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn

def train(trainloader, testloader, tvae, optimizer, args, writer):
    logger = Logger(name='logs', base=args.root)
    best_loss = 11e8
    
    # evaluate the init
    test_KLD = utils.MovingMetric()
    test_recon_loss = utils.MovingMetric()
    test_loss = utils.MovingMetric()

    for i, x in enumerate(testloader):
        x = x.to(tvae.device)
        z, u, s, x_recon, kl_z, kl_u, recon_loss = tvae(x)
        
        avg_KLD = (kl_z.sum() + kl_u.sum()) / x.shape[0]
        recon_loss = recon_loss.sum() / x.shape[0]
        loss = recon_loss + avg_KLD

        test_KLD.add(avg_KLD.item()*x.shape[0], x.shape[0])
        test_recon_loss.add(recon_loss.item()*x.shape[0], x.shape[0])
        test_loss.add(loss.item()*x.shape[0], x.shape[0])

    test_KLD = test_KLD.get_val()
    test_recon_loss = test_recon_loss.get_val()
    test_loss = test_loss.get_val()

    logger.add_scalar(0, 'test_KLD', test_KLD)
    logger.add_scalar(0, 'test_recon_loss', test_recon_loss)
    logger.add_scalar(0, 'test_loss', test_loss)
    logger.iter_info()
    logger.save()
    writer.add_scalar('test/KLD',test_KLD,0)
    writer.add_scalar('test/recon_loss',test_recon_loss,0)
    writer.add_scalar('test/loss',test_loss,0)

    torch.save(tvae.state_dict(), os.path.join(args.root, 'tvae_params_init.torch'))

    for epoch in range(1, args.num_epochs + 1):

        #set learning rate
        adjust_learning_rate(optimizer, lr_linear(epoch-1))

        train_KLD = utils.MovingMetric()
        train_recon_loss = utils.MovingMetric()
        train_loss = utils.MovingMetric()

        for i, x in enumerate(trainloader):
            optimizer.zero_grad()
            x = x.to(tvae.device)
            z, u, s, x_recon, kl_z, kl_u, recon_loss = tvae(x) 
            
            avg_KLD = (kl_z.sum() + kl_u.sum()) / x.shape[0]
            recon_loss = recon_loss.sum() / x.shape[0]
            loss = recon_loss + avg_KLD

            loss.backward()
            optimizer.step()

            train_KLD.add(avg_KLD.item()*x.shape[0], x.shape[0])
            train_recon_loss.add(recon_loss.item()*x.shape[0], x.shape[0])
            train_loss.add(loss.item()*x.shape[0], x.shape[0])

        test_KLD = utils.MovingMetric()
        test_recon_loss = utils.MovingMetric()
        test_loss = utils.MovingMetric()

        for i, x in enumerate(testloader):
            x = x.to(tvae.device)
            z, u, s, x_recon, kl_z, kl_u, recon_loss = tvae(x)
           
            avg_KLD = (kl_z.sum() + kl_u.sum()) / x.shape[0]
            recon_loss = recon_loss.sum() / x.shape[0]
            loss = recon_loss + avg_KLD

            test_KLD.add(avg_KLD.item()*x.shape[0], x.shape[0])
            test_recon_loss.add(recon_loss.item()*x.shape[0], x.shape[0])
            test_loss.add(loss.item()*x.shape[0], x.shape[0])
        
        train_KLD = train_KLD.get_val()
        train_recon_loss = train_recon_loss.get_val()
        train_loss = train_loss.get_val()
        test_KLD = test_KLD.get_val()
        test_recon_loss = test_recon_loss.get_val()
        test_loss = test_loss.get_val()

        logger.add_scalar(epoch, 'train_KLD', train_KLD)
        logger.add_scalar(epoch, 'train_recon_loss', train_recon_loss)
        logger.add_scalar(epoch, 'train_loss', train_loss)
        logger.add_scalar(epoch, 'test_KLD', test_KLD)
        logger.add_scalar(epoch, 'test_recon_loss', test_recon_loss)
        logger.add_scalar(epoch, 'test_loss', test_loss)
        logger.iter_info()
        logger.save()

        writer.add_scalar('train/KLD',train_KLD,epoch)
        writer.add_scalar('train/recon_loss',train_recon_loss,epoch)
        writer.add_scalar('train/loss',train_loss,epoch)
        writer.add_scalar('test/KLD',test_KLD,epoch)
        writer.add_scalar('test/recon_loss',test_recon_loss,epoch)
        writer.add_scalar('test/loss',test_loss,epoch)

        is_best = (test_loss < best_loss)
        if is_best:
            best_loss = test_loss
            torch.save(tvae.state_dict(), os.path.join(args.root, 'tvae_params.torch'))   
            if args.add_save_path : 
                torch.save(tvae.state_dict(), os.path.join(args.add_save_path, 'tvae_params.torch'))  

    torch.save(tvae.state_dict(), os.path.join(args.root, 'tvae_params_lastepoch.torch'))
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
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--sgd_momentum',type=float, default=0.9)
    
    #evaluation
    parser.add_argument('--test_bs', default=512, type=int)

    #model specifics
    parser.add_argument('--n_caps', default=5, type=int)
    parser.add_argument('--cap_dim', default=4, type=int)
    parser.add_argument('--mu_init', default=30, type=int)
    
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

    #initialise model; some model parameters are set here, because they likely 
    # wont be changed during experimentation
    n_cin = 1
    n_hw = 3 
    group_kernel = (1,3,1)
    n_transforms = 1 
    n_caps = args.n_caps 
    cap_dim = args.cap_dim
    s_dim = n_caps * cap_dim
    mu_init = args.mu_init 
    z_encoder = Gaussian_Encoder(MLP_Encoder(s_dim=s_dim, n_cin=n_cin, n_hw=n_hw),
                                    loc=0.0, scale=1.0)

    u_encoder = Gaussian_Encoder(MLP_Encoder(s_dim=s_dim, n_cin=n_cin, n_hw=n_hw),                                
                                    loc=0.0, scale=1.0)

    decoder = Gaussian_Decoder(MLP_Decoder(s_dim=s_dim, n_cout=n_cin, n_hw=n_hw))

    grouper = Stationary_Capsules_1d(
                        nn.ConvTranspose3d(in_channels=1, out_channels=1,
                                            kernel_size=group_kernel, 
                                            padding=(2*(group_kernel[0] // 2), 
                                                    2*(group_kernel[1] // 2),
                                                    2*(group_kernel[2] // 2)),
                                            stride=(1,1,1), padding_mode='zeros', bias=False),
                        lambda x: F.pad(x, (group_kernel[2] // 2, group_kernel[2] // 2,
                                            group_kernel[1] // 2, group_kernel[1] // 2,
                                            group_kernel[0] // 2, group_kernel[0] // 2), 
                                            mode='circular'),
                    n_caps=n_caps, cap_dim=cap_dim, n_transforms=n_transforms,
                    mu_init=mu_init, device=device)

    tvae = TVAE(z_encoder, u_encoder, decoder, grouper)
    
    #configure optimisation
    optimizer = torch.optim.SGD(tvae.parameters(), lr=args.lr, momentum= args.sgd_momentum, 
                                    weight_decay=args.weight_decay)

    writer = SummaryWriter(args.root)

    train(trainloader, testloader, tvae, optimizer, args, writer)

    writer.flush()
