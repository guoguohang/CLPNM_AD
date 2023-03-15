import yaml
from torch.utils.data.sampler import Sampler
import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
import math


def read_config_file(config_parser):
    args = config_parser.parse_args()
    config_file_path = 'config_files/' + args.config_file
    with open(config_file_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        config_parser.set_defaults(**cfg)
    args = config_parser.parse_args()
    return args


def init_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.scheduler['lr']
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class IndSampler(Sampler):
    def __init__(self, idx_list):
        self.indexes = idx_list

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=100, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_model = None
        self.best_epoch = 0
        self.detect_res = None

    def __call__(self, val_loss, model, epoch, detect_res):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = model
            self.best_epoch = epoch
            self.detect_res = detect_res
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model
            self.best_epoch = epoch
            self.detect_res = detect_res
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
