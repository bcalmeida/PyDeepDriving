# from fastai.imports import *
# from fastai.torch_imports import *
# from fastai.transforms import *
# from fastai.conv_learner import *
# from fastai.model import *
# from fastai.dataset import *
# from fastai.sgdr import *
# from fastai.plots import *

from torch import nn
from torch.nn.init import kaiming_normal
from torchvision.models import resnet34
from fastai.layers import AdaptiveConcatPool2d, Flatten
from fastai.initializers import apply_init
from fastai.core import set_trainable

class CNNtoFC(nn.Module):
    def __init__(self):
        super().__init__()
        
        resnet_model = resnet34(pretrained=True)
        encoder_layers = list(resnet_model.children())[:8] + [AdaptiveConcatPool2d(), Flatten()]
        self.encoder = nn.Sequential(*encoder_layers).cuda()
        for param in self.encoder.parameters():
            param.requires_grad = False
        set_trainable(self.encoder, False) # fastai fit bug
        
        self.linear = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512,14)).cuda()
        apply_init(self.linear, kaiming_normal)
        set_trainable(self.linear, True) # fastai fit bug
        
    def forward(self, x):
        encodes = self.encoder(x) # shape (seq_len*bs, 1024)
        return self.linear(encodes) # shape (seq_len*bs, 14)
