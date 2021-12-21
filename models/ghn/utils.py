import torch 
import numpy as np 

def capacity(model):
    c, n = 0, 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            c += 1
            n += np.prod(p.shape)
    return c, int(n)
    
def default_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'