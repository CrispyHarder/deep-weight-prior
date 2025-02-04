import torch
import utils
import numpy as np
import os
import time
from models.cifar import ResNet
import utils
from logger import Logger
import myexman
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from my_utils import MultistepMultiGammaLR

def predict(data, net):
    pred = []
    l = []
    for x, y in data:
        l.append(y.numpy())
        x = x.to(device)
        p = F.log_softmax(net(x), dim=1)
        pred.append(p.data.cpu().numpy())
    return np.concatenate(pred), np.concatenate(l)

def eval_step(e,net,valloader,logger,writer,args,opt,train_acc,train_nll,lrscheduler,step=True):
    global t0
    global best_acc

    net.eval()
    with torch.no_grad():
        logp_val, labels = predict(valloader, net)
        val_acc = np.mean(logp_val.argmax(1) == labels)
        val_nll = -logp_val[np.arange(len(labels)), labels].mean()

    logger.add_scalar(e, 'train_nll', train_nll.get_val())
    logger.add_scalar(e, 'train_acc', train_acc.get_val())
    logger.add_scalar(e, 'val_nll', val_nll)
    logger.add_scalar(e, 'val_acc', val_acc)
    logger.add_scalar(e, 'lr', opt.param_groups[0]['lr'])
    logger.add_scalar(e, 'sec', time.time() - t0)
    logger.iter_info()
    logger.save()

    writer.add_scalar( 'train/nll', train_nll.get_val(),e)
    writer.add_scalar( 'val/nll', val_nll,e)
    writer.add_scalar( 'train/acc', train_acc.get_val(),e)
    writer.add_scalar( 'val/acc', val_acc,e)
    writer.add_scalar( 'lr', opt.param_groups[0]['lr'],e)
    writer.add_scalar( 'sec', time.time() - t0,e)
    
    is_best = best_acc < val_acc
    if is_best:
        best_acc = val_acc
        torch.save(net.state_dict(), os.path.join(args.root, 'net_params.torch'))     

    t0 = time.time()
    if step:
        lrscheduler.step()
      
parser = myexman.ExParser(file=__file__)
#general settings
parser.add_argument('--name', default='')
parser.add_argument('--gpu_id', default='0')
parser.add_argument('--seed', default=5743, type=int)
parser.add_argument('--epochs', default=120, type=int, help='Number of epochs')
parser.add_argument('--bs', default=128, type=int, help='Batch size')
parser.add_argument('--test_bs', default=500, type=int, help='Batch size for test dataloader')

#model init settings MULTI init (if used, single init is ignored)
parser.add_argument('--mult_init_mode', default= 'he', type = str,
                    help = '''such as vqvae1.3''')
parser.add_argument('--mult_init_root', type=str, default=os.path.join('data','resnet20_cifar','3x3'))
parser.add_argument('--mult_init_prior', type=str, default='',
                    help='''such as pixelcnn0''')

#optimizer settings
parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--milestones', type=int, nargs='*', default=[80,100])
parser.add_argument('--gammas', default=[0.5,0.2], nargs='*', type=float)


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

fmt = {
    'lr': '.5f',
    'sec': '.0f',
}
logger = Logger('logs', base=args.root, fmt=fmt)

# Load Datasets
t0 = time.time()
trainloader, valloader, _ = utils.load_cifar10_loaders(args.bs, args.test_bs)

#get model
net = ResNet([3,3,3],num_classes=10).to(device)

# Initialization
net.mult_weights_init(args.mult_init_mode, args.mult_init_root, device=device, prior=args.mult_init_prior)

opt = torch.optim.SGD(net.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
lrscheduler = MultistepMultiGammaLR(opt, milestones=args.milestones, 
                                    gamma=args.gammas)

t0 = time.time()


#add a tensorboard writer
writer = SummaryWriter(args.root)
best_acc = 0.

#for init eval 
train_acc = utils.MovingMetric()
train_nll = utils.MovingMetric()
eval_step(0,net,valloader,logger,writer,args,opt,train_acc,train_nll,lrscheduler,step=False)

for e in range(1, args.epochs + 1):
    net.train()
    train_acc = utils.MovingMetric()
    train_nll = utils.MovingMetric()

    for x, y in trainloader:
        opt.zero_grad()

        x = x.to(device)
        y = y.to(device)

        p = net(x)

        nll = F.cross_entropy(p, y)

        nll.backward()

        opt.step()

        acc = torch.sum(p.max(1)[1] == y)
        train_acc.add(acc.item(), p.size(0))
        train_nll.add(nll.item() * x.size(0), x.size(0))
    
    # after one train epoch, do eval step 
    eval_step(e,net,valloader,logger,writer,args,opt,train_acc,train_nll,lrscheduler)

torch.save(net.state_dict(), os.path.join(args.root, 'net_params_lastepoch.torch'))
torch.save(opt.state_dict(), os.path.join(args.root, 'opt_params_lastepoch.torch'))

writer.flush()

        
