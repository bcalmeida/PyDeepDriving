# from fastai.imports import *
# from fastai.torch_imports import *
# from fastai.transforms import *
# from fastai.conv_learner import *
# from fastai.model import *
# from fastai.dataset import *
# from fastai.sgdr import *
# from fastai.plots import *

import math
import torch
from fastai.core import T

# Metrics
# preds is a torch array, do not use np.mean() or other numpy functions
# Now written in a way that accepts both np arrays and torch tensors
def rmse(preds,y):
    return math.sqrt(((preds-y)**2).mean())

def mae(preds,y):
    #return abs(preds-y).mean()
    return torch.mean(torch.abs(preds-y))

def mae_idx(preds, y, idx):
    #return abs(preds[:,idx]-y[:,idx]).mean()
    return torch.mean(torch.abs((preds[:,idx]-y[:,idx])))

def mae_angle(preds, y):  return mae_idx(preds, y, 0)
def mae_toM_L(preds, y):  return mae_idx(preds, y, 1)
def mae_toM_M(preds, y):  return mae_idx(preds, y, 2)
def mae_toM_R(preds, y):  return mae_idx(preds, y, 3)
def mae_d_L(preds, y):    return mae_idx(preds, y, 4)
def mae_d_R(preds, y):    return mae_idx(preds, y, 5)
def mae_toM_LL(preds, y): return mae_idx(preds, y, 6)
def mae_toM_ML(preds, y): return mae_idx(preds, y, 7)
def mae_toM_MR(preds, y): return mae_idx(preds, y, 8)
def mae_toM_RR(preds, y): return mae_idx(preds, y, 9)
def mae_d_LL(preds, y):   return mae_idx(preds, y, 10)
def mae_d_MM(preds, y):   return mae_idx(preds, y, 11)
def mae_d_RR(preds, y):   return mae_idx(preds, y, 12)
def mae_fast(preds, y):   return mae_idx(preds, y, 13)
def acc_fast(preds, y):
    idx = 13
    preds = (preds[:, idx] > 0.5)
    y = (y[:, idx] > 0.5)
    return (preds == y).float().mean()

METRICS = [
    rmse,
    mae,
    mae_angle,
    mae_toM_L,
    mae_toM_M,
    mae_toM_R,
    mae_d_L,
    mae_d_R,
    mae_toM_LL,
    mae_toM_ML,
    mae_toM_MR,
    mae_toM_RR,
    mae_d_LL,
    mae_d_MM,
    mae_d_RR,
    mae_fast,
    acc_fast,
]

def print_all_metrics(preds, y):
    preds = T(preds)
    y = T(y)
    for f in METRICS:
        print("%-12s %.3f" % (f.__name__, f(preds, y)))

def calc_all_metrics(preds, y):
    preds = T(preds)
    y = T(y)
    res = []
    for f in METRICS:
        res.append(f(preds, y))
    return res

def print_calced_metrics(*mets):
    header = ""
    lines = []
    for f in METRICS:
        header += "%-11s" % f.__name__
    for met in mets:
        line = ""
        for v in met:
            line += "%-11.3f" % v
        lines.append(line)
    
    print(header)
    for line in lines:
        print(line)
