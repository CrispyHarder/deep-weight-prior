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
    def __init__(self,init,device,ds,dataloaders):
        self.device = device
        self.ds = ds
        self.init = init 
        self.dataloaders = dataloaders 
        self.poss_runs = []
        self.poss_models = []
        self.pred_all_ds = []
        self.labels_all_ds = []
        self.search_models()
        self.get_models()
        self.get_predictions()
        
    
    # def predict(self,x):
    #     x = x.to(self.device)
    #     logits = 0.
    #     with torch.no_grad():
    #         for member in self.members:
    #             logits += member(x)
    #     return logits/self.n_members

    def search_models(self):
        path = os.path.join('logs',f'exman-train-net-{self.ds}.py','runs')
        for run in os.listdir(path):
            yaml_p = os.path.join(path,run,'params.yaml')
            with open(yaml_p) as f:
                dict = yaml.full_load(f)
            if not 'mult_init_prior' in dict:
                if dict['mult_init_mode'] == self.init[0]:
                    self.poss_runs.append(os.path.join(path,run))
            elif 'mult_init_prior' in dict and len(self.init) ==1:
                if dict['mult_init_mode'] == self.init[0] and dict['mult_init_prior'] == '':
                    self.poss_runs.append(os.path.join(path,run))
            elif 'mult_init_prior' in dict and len(self.init) ==2:
                if dict['mult_init_mode'] == self.init[0] and dict['mult_init_prior'] == self.init[1]:
                    self.poss_runs.append(os.path.join(path,run))
            if dict['mult_init_mode'] == self.init:
                self.poss_runs.append(os.path.join(path,run))

    def get_models(self):
        for run_id in self.poss_runs:
            model = resnet.resnet20()
            model.load_state_dict(torch.load(os.path.join(run_id,'net_params.torch'),map_location=self.device))
            model = model.to(self.device)
            model.eval()
            self.poss_models.append(model)

    def get_predictions(self):
        for dl in self.dataloaders: 
            predictions_one_ds = []
            for i,model in enumerate(self.poss_models):
                pred,labels = self.predict(dl,model)
                predictions_one_ds.append(pred)
                if i == 0:
                    self.labels_all_ds.append(labels)
            self.pred_all_ds.append(np.array(predictions_one_ds))
        
    def get_ens_prediction(self,n_members,corr_level):
        sampled_ind = np.random.randint(0,25,size=n_members)
        pred = self.pred_all_ds[corr_level][sampled_ind]
        labels = np.array(self.labels_all_ds[corr_level])
        pred = np.mean(pred,axis=0)
        prob = F.softmax(torch.from_numpy(pred),dim=1).numpy()
        return pred,prob,labels




    # def sample_ensemble(self):
    #     sampled_ind = np.random.randint(0,25,size=self.n_members)
    #     run_ids = self.poss_runs[sampled_ind]
    #     for run_id in run_ids:
    #         model = resnet.resnet20()
    #         model.load_state_dict(torch.load(os.path.join(run_id,'net_params.torch'),map_location=self.device))
    #         model = model.to(self.device)
    #         model.eval()
    #         self.members.append(model)
    
    def predict(self,data, net):
        prob = []
        pred = []
        l = []
        for i,(x, y) in enumerate(data):
            l.append(y.numpy())
            x = x.to(self.device)
            p = net(x)
            pred.append(p.data.cpu().numpy())
        return np.concatenate(pred), np.concatenate(l)


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
