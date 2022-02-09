import torch 
import math     
from torch import nn 
from torch.nn import functional as F
from models.cifar import resnet
import yaml
import os 
import re
import warnings
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np 


class MultistepMultiGammaLR(_LRScheduler):
    """Decays the learning rate of each parameter group by a different gamma,
    (which can be different every time in this adaption of MultiStepLR) once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (list(float)): Multiplicative factor of learning rate decay.
            One for every milestone
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example: with same gamma every time 
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    def __init__(self, optimizer, milestones, gamma, last_epoch=-1, verbose=False):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.counter = -1
        super(MultistepMultiGammaLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        
        if not self._get_lr_called_within_step: # pylint: disable=no-member
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        #if we change lr, increase counter by one
        self.counter += 1
        return [group['lr'] * self.gamma[self.counter]
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        lr_modifier_total = 1
        for i in range(self.counter):
            lr_modifier_total = lr_modifier_total * self.gamma[i]
        return [base_lr * self.gamma ** lr_modifier_total
                for base_lr in self.base_lrs]
                

def get_state_dict_from_checkpoint(checkpoint_path, map_location=None):
    '''loads the state dict from a given checkpoint path'''
    if map_location:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    elif torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    return checkpoint['state_dict']

def load_statedict(path, map_location=None):
    if map_location:
        state_dict = torch.load(path, map_location=map_location)
    elif torch.cuda.is_available():
        state_dict = torch.load(path)
    else:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
    return state_dict 

def load_resnet_from_checkpoint(checkpoint_path, model_type, dataset_name):
    '''Gets a path to a checkpoint and a model type and loads the model 
    using the state dict'''

    model = resnet.__dict__[model_type]()
    model.load_state_dict(get_state_dict_from_checkpoint(checkpoint_path))
    if torch.cuda.is_available():
        model.cuda()  
    return model

def save_args_params(args, dir_path):
    dumbd = args.__dict__.copy()
    path = os.path.join(dir_path,'params.yaml')
    if os.path.exists(path):
        os.remove(path)
    with open(path,'a') as f:
        yaml.dump(dumbd,f)

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def brier_multi(targets, probs):
    '''Takes in a 1 dim tensor of targets and a 2-dim vector of probability scores and 
    computes the Brier score'''
    targets = to_one_hot(targets)
    return np.mean(np.sum((probs - targets)**2, axis=1))

def to_one_hot (labels,num_classes=10):
    n = np.shape(labels)[0]
    one_hot = np.zeros((n,num_classes))
    one_hot[np.arange(n),labels] = 1 
    return one_hot

class Ensemble():
    def __init__(self,init,n_members,device):
        self.device = device
        self.members = []
        self.find_models(init,n_members)
        self.n_members = n_members
    
    def predict(self,x):
        x = x.to(self.device)
        logits = 0.
        with torch.no_grad():
            for member in self.members:
                logits += member(x)
        return logits/self.n_members

    def find_models(self,init,n_members):
        found_members = 0
        path = os.path.join('logs','exman-train-net.py','runs')
        for run in os.listdir(path)[2:]:
            yaml_p = os.path.join(path,run,'params.yaml')
            with open(yaml_p) as f:
                dict = yaml.full_load(f)
            if dict['mult_init_mode'] == init:
                found_members += 1 
                model = resnet.resnet20()
                try:
                    model.load_state_dict(torch.load(os.path.join(path,run, 'vae_params_lastepoch.torch'),map_location=self.device))
                except:
                    model.load_state_dict(torch.load(os.path.join(path,run, 'net_params_lastepoch.torch'),map_location=self.device))
                model = model.to(self.device)
                model.eval()
                self.members.append(model)
                if found_members == n_members:
                    return
        raise RuntimeError


# Some keys used for the following dictionaries
# from https://github.com/torrvision/focal_calibration/blob/main/Metrics/metrics.py

COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'


def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0


def _populate_bins(confs, preds, labels, num_bins=10):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + \
            (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(
                bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / \
                float(bin_dict[binn][COUNT])
    return bin_dict


def expected_calibration_error(confs, preds, labels, num_bins=10):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * \
            abs(bin_accuracy - bin_confidence)
    return ece
