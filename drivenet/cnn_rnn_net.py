# from fastai.imports import *
# from fastai.torch_imports import *
# from fastai.transforms import *
# from fastai.conv_learner import *
# from fastai.model import *
# from fastai.dataset import *
# from fastai.sgdr import *
# from fastai.plots import *

import torch
from torch import nn
from torch.autograd import Variable
from torchvision.models import resnet34
from fastai.layers import AdaptiveConcatPool2d, Flatten
from fastai.initializers import apply_init
from fastai.core import set_trainable, T

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

class CNNtoRNNFeedback(nn.Module):
    def __init__(self, encode_size, hidden_size, num_layers, seq_len, bs, output_size, use_ground_truth):
        super().__init__()
        self.encode_size = encode_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.bs = bs
        self.output_size = output_size
        
        resnet_model = resnet34(pretrained=True)
        encoder_layers = list(resnet_model.children())[:8] + [AdaptiveConcatPool2d(), Flatten()]
        self.encoder = nn.Sequential(*encoder_layers).cuda()
        for param in self.encoder.parameters():
            param.requires_grad = False
        set_trainable(self.encoder, False) # fastai fit bug
        
        self.lstm = nn.LSTM(encode_size+output_size, hidden_size, num_layers).cuda()
        self.h, self.c = self.init_hidden()
        set_trainable(self.lstm, True) # fastai fit bug
        
        self.linear = nn.Linear(hidden_size, output_size).cuda()
        set_trainable(self.linear, True) # fastai fit bug
        
        self.init_weights() # added nb-23
        
        self.use_ground_truth = use_ground_truth
        self.last_pred = Variable(torch.zeros(seq_len*bs, output_size).cuda())
        
    # added nb-23
    def init_weights(self):
        for layer_weights in self.lstm.all_weights:
            weight_ih, weight_hh, bias_ih, bias_hh = layer_weights
            nn.init.xavier_normal(weight_ih)
            nn.init.xavier_normal(weight_hh)
            bias_ih.data.fill_(0.)
            bias_hh.data.fill_(0.)
        apply_init(self.linear, nn.init.kaiming_normal)
        
    def init_hidden(self):
        # print("hidden state initialized")
        return (Variable(T(torch.zeros(self.num_layers, self.bs, self.hidden_size))),
                Variable(T(torch.zeros(self.num_layers, self.bs, self.hidden_size)))) # T to put in gpu
        
    def forward(self, x, y_prev):
        # x has shape (seq_len*bs, *img_shape), and was previously windowed (seq_len, bs, *img_shape)
        # y_prev has shape (seq_len*bs, output_shape), and was previously windowed (seq_len, bs, output_shape)
        if self.use_ground_truth and self.training: # Ground truth may only be used on training
            y_context = y_prev
        else:
            y_context = self.last_pred
        
        encodes = self.encoder(x) # shape (seq_len*bs, 1024)
        encodes_and_context = torch.cat((encodes, y_context), dim=1) # shape (seq_len*bs, encode_size+output_size)
        encodes_and_context = encodes_and_context.view(-1, self.bs, self.encode_size+self.output_size) # shape (seq_len', bs, enc+out)
        
        output, (self.h, self.c) = self.lstm(encodes_and_context, (self.h, self.c)) # output shape (seq_len, bs, hidden_size)
        output = output.view(self.seq_len*self.bs, self.hidden_size)
        
        # truncated backprop
        self.h = repackage_hidden(self.h)
        self.c = repackage_hidden(self.c)
        
        # y shape (seq_len*bs, output_size)
        self.last_pred = self.linear(output)
        return self.last_pred
    
    def reset(self):
        self.h, self.c = self.init_hidden()