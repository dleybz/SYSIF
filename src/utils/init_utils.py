import random
import torch
import logging

def init_random_seed(seed):    
    # init random seed
    if seed == -1:
        random_seed = random.randint(0, 99999)
    else:
        random_seed = seed
    
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    return random_seed

def init_device(device):
    # init device
    device=torch.device(device)
    if device == 'cuda':
        n_gpu = torch.cuda.device_count()
        if n_gpu == 0:
            logging.warning('No GPU found! exit!')
        logging.info('# GPUs: %d'%n_gpu)