import os
import numpy as np
import torch
import torch.nn.functional as F
import utils
from logger import Logger
from models import vqvae1,pixelcnn
import myexman
from my_utils import save_args_params


def train(trainloader, testloader, pixelcnn, optimizer, scheduler, args):
    logger = Logger(name='logs', base=args.root)
    best_loss = 1000
    for epoch in range(1, args.num_epochs + 1):
        adjust_learning_rate(optimizer, lr_linear(epoch))
        scheduler.step()
        train_loss = utils.MovingMetric()
        test_loss = utils.MovingMetric()

        for i,latents in enumerate(trainloader):
            logits = pixelcnn(latents)
            logits = logits.permute(0,2,3,1).contiguous() 

            optimizer.zero_grad()
            loss = F.cross_entropy(logits.view(-1, args.k),latents.view(-1))
            loss.backward()
            optimizer.step()

            train_loss.add(loss.item(), latents.size(0))

        for i,latents in enumerate(testloader):
            with torch.no_grad():
                logits = pixelcnn(latents)
                logits = logits.permute(0,2,3,1).contiguous() 
                t_loss = F.cross_entropy(logits.view(-1, args.k),latents.view(-1))

            test_loss.add(t_loss.item(), latents.size(0))

        train_loss = train_loss.get_val()
        test_loss = test_loss.get_val()

        logger.add_scalar(epoch, 'train_loss', train_loss)
        logger.add_scalar(epoch, 'test_loss', test_loss)

        logger.iter_info()
        logger.save()

        if (epoch-1) % 10 == 0:
            torch.save(pixelcnn.state_dict() , os.path.join(args.root, 'pixelcnn_params_epoch_{}.torch'.format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(args.root, 'opt_params_epoch_{}.torch'.format(epoch)))

        is_best = (epoch == 1) or (test_loss < best_loss)
        if is_best:
            best_loss = test_loss
            torch.save(pixelcnn.state_dict(), os.path.join(args.root, 'pixelcnn_params.torch'))   
            if args.add_save_path : 
                torch.save(pixelcnn.state_dict(), os.path.join(args.add_save_path, 'pixel_params.torch'))  
   

    torch.save(pixelcnn.state_dict(), os.path.join(args.root, 'pixelcnn_params_lastepoch.torch'))
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

    #general training information 
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu_id', default='0')

    #for eval (outputs)
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--test_bs', default=512, type=int)
    parser.add_argument('--verbose', default=1, type=int)

    #model spec 
    parser.add_argument('--input_dim', default=8, type=int,
                        help='''the size of the codebook of the vqvae''')
    parser.add_argument('--dim', default=16, type=int,
                        help='''the hidden size (number of channels per layer)''')
    parser.add_argument('--n_layers', type=int, default=10)


    #introduced for saving the models twice
    parser.add_argument('--add_save_path',default='')


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

    pixel = pixelcnn.GatedPixelCNN(input_dim=args.input_dim,dim=args.dim,n_layers=args.n_layers)

    optimizer = torch.optim.Adam(pixel.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.decay)

    train(trainloader, testloader, pixel, optimizer, scheduler, args)