import os
import numpy as np
import torch
import utils
from logger import Logger
import myexman
from my_utils import save_args_params
from torch.utils.tensorboard import SummaryWriter
from models.lvae import LVAE
import torch.nn.functional as F

def train(trainloader, testloader, lvae, optimizer, args, writer):
    logger = Logger(name='logs', base=args.root)
    best_loss = 100000000
    
    # evaluate the init
    test_KLD = utils.MovingMetric()
    test_recon_loss = utils.MovingMetric()
    test_loss = utils.MovingMetric()

    for i, x in enumerate(testloader):
        x = x.to(lvae.device)
        _ , recon_loss, kl_loss = lvae(x)
        
        loss = recon_loss + kl_loss

        test_KLD.add(kl_loss.item()*x.size(0), x.size(0))
        test_recon_loss.add(recon_loss.item()*x.size(0), x.size(0))
        test_loss.add(loss.item()*x.size(0), x.size(0))

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

    torch.save(lvae.state_dict(), os.path.join(args.root, 'lvae_params_init.torch'))

    for epoch in range(1, args.num_epochs + 1):

        #set learning rate
        adjust_learning_rate(optimizer, lr_linear(epoch-1))

        train_KLD = utils.MovingMetric()
        train_recon_loss = utils.MovingMetric()
        train_loss = utils.MovingMetric()

        for i, x in enumerate(trainloader):
            optimizer.zero_grad()
            x = x.to(lvae.device)
            _ , recon_loss, kl_loss = lvae(x) 
            
            avg_KLD = kl_loss
            loss = recon_loss + avg_KLD

            loss.backward()
            optimizer.step()

            train_KLD.add(avg_KLD.item()*x.size(0), x.size(0))
            train_recon_loss.add(recon_loss.item()*x.size(0), x.size(0))
            train_loss.add(loss.item()*x.size(0), x.size(0))

        test_KLD = utils.MovingMetric()
        test_recon_loss = utils.MovingMetric()
        test_loss = utils.MovingMetric()

        for i, x in enumerate(testloader):
            x = x.to(lvae.device)
            _ , recon_loss, kl_loss = lvae(x)
           
            avg_KLD = kl_loss
            loss = recon_loss + avg_KLD

            test_KLD.add(avg_KLD.item()*x.size(0), x.size(0))
            test_recon_loss.add(recon_loss.item()*x.size(0), x.size(0))
            test_loss.add(loss.item()*x.size(0), x.size(0))
        
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
            torch.save(lvae.state_dict(), os.path.join(args.root, 'lvae_params.torch'))   
            if args.add_save_path : 
                torch.save(lvae.state_dict(), os.path.join(args.add_save_path, 'lvae_params.torch'))  

    torch.save(lvae.state_dict(), os.path.join(args.root, 'lvae_params_lastepoch.torch'))
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
    
    #evaluation
    parser.add_argument('--test_bs', default=512, type=int)

    #model specifics
    parser.add_argument('--dims', default=[100,50,10], nargs='*', type=float)
    parser.add_argument('--z_dim', default=2, type=int)
    
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

    dims = [int(el) for el in args.dims]
    lvae = LVAE(dims,args.z_dim,device)
    
    #configure optimisation
    optimizer = torch.optim.Adam(lvae.parameters(), lr=args.lr, weight_decay=args.weight_decay) 

    writer = SummaryWriter(args.root)

    train(trainloader, testloader, lvae, optimizer, args, writer)

    writer.flush()
