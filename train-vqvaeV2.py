import os
import numpy as np
import torch
import utils
from logger import Logger
import myexman
from my_utils import save_args_params
from models import vqvae1
from torch.utils.tensorboard import SummaryWriter
import torch.distributions as dist
import torch.nn.functional as F

def train(trainloader, testloader, vqvae, optimizer, args, writer):
    logger = Logger(name='logs', base=args.root)
    best_loss = 100000000
    
    # evaluate the init
    test_recon_loss = utils.MovingMetric()
    test_vq_loss = utils.MovingMetric()
    test_perplexity = utils.MovingMetric()
    test_loss = utils.MovingMetric()

    for i, x in enumerate(testloader):
        x = x.to(vqvae.device)

        vq_loss, x_recon, perplexity = vqvae(x)
        recon_loss = F.mse_loss(x,x_recon)
        loss = vq_loss + recon_loss

        test_recon_loss.add(recon_loss.item()*x.size(0), x.size(0))
        test_vq_loss.add(vq_loss.item()*x.size(0), x.size(0))
        test_perplexity.add(perplexity.item(), x.size(0))
        test_loss.add(loss.item()*x.size(0), x.size(0))

    test_recon_loss = test_recon_loss.get_val()
    test_vq_loss = test_vq_loss.get_val()
    test_perplexity = test_perplexity.get_val()
    test_loss = test_loss.get_val()

    logger.add_scalar(0, 'test_recon_loss', test_recon_loss)
    logger.add_scalar(0, 'test_vq_loss', test_vq_loss)
    logger.add_scalar(0, 'test_perplexity', test_perplexity)
    logger.add_scalar(0, 'test_loss', test_loss)
    logger.iter_info()
    logger.save()
    writer.add_scalar('test/recon_loss',test_recon_loss,0)
    writer.add_scalar('test/vq_loss',test_vq_loss,0)
    writer.add_scalar('test/perplexity',test_perplexity,0)
    writer.add_scalar('test/loss',test_loss,0)

    torch.save(vqvae.state_dict(), os.path.join(args.root, 'vqvae_params_init.torch'))

    for epoch in range(1, args.num_epochs + 1):

        #set learning rate
        adjust_learning_rate(optimizer, lr_linear(epoch-1))

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

            train_recon_loss.add(recon_loss.item()*x.size(0), x.size(0))
            train_vq_loss.add(vq_loss.item()*x.size(0), x.size(0))
            train_perplexity.add(perplexity.item(), x.size(0))
            train_loss.add(loss.item()*x.size(0), x.size(0))

        test_recon_loss = utils.MovingMetric()
        test_vq_loss = utils.MovingMetric()
        test_perplexity = utils.MovingMetric()
        test_loss = utils.MovingMetric()

        for i, x in enumerate(testloader):
            x = x.to(vqvae.device)

            vq_loss, x_recon, perplexity = vqvae(x)
            recon_loss = F.mse_loss(x,x_recon)
            loss = vq_loss + recon_loss

            test_recon_loss.add(recon_loss.item()*x.size(0), x.size(0))
            test_vq_loss.add(vq_loss.item()*x.size(0), x.size(0))
            test_perplexity.add(perplexity.item(), x.size(0))
            test_loss.add(loss.item()*x.size(0), x.size(0))

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

        writer.add_scalar('train/recon_loss',train_recon_loss,epoch)
        writer.add_scalar('train/vq_loss',train_vq_loss,epoch)
        writer.add_scalar('train/perplexity',train_perplexity,epoch)
        writer.add_scalar('train/loss',train_loss,epoch)
        writer.add_scalar('test/recon_loss',test_recon_loss,epoch)
        writer.add_scalar('test/vq_loss',test_vq_loss,epoch)
        writer.add_scalar('test/perplexity',test_perplexity,epoch)
        writer.add_scalar('test/loss',test_loss,epoch)

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
    parser.add_argument('--z_dim', default=8, type=int)
    parser.add_argument('--hidden_dim', default=16, type=int)
    parser.add_argument('--num_embeddings', type=int, default=128)
    parser.add_argument('--ema_decay', type= float, default=0.)
    parser.add_argument('--commitment_cost', type=float, default=0.25)
    
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

    decoder = vqvae1.Decoder3x3(args.z_dim, args.hidden_dim)
    encoder = vqvae1.Encoder3x3(args.z_dim, args.hidden_dim)
    vqvae = vqvae1.VQVAE(encoder, decoder, args.num_embeddings, args.commitment_cost, 
            device=device, decay=args.ema_decay)
    
    #configure optimisation
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    writer = SummaryWriter(args.root)

    train(trainloader, testloader, vqvae, optimizer, args, writer)

    writer.flush()
