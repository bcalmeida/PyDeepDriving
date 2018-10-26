# from fastai.imports import *
# from fastai.torch_imports import *

import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from fastai.dataset import open_image

def batchify(data, bs):
    nbatch = data.shape[0] // bs
    splits = np.split(data[:nbatch*bs], bs)
    return np.stack(splits, axis=1)

class BatchifiedDataset(Dataset):
    def __init__(self, data, bs, seq_len, transform, PATH, IMAGES_FOLDER):
        bdata = batchify(data, bs)
        self.bdata = bdata.reshape(bdata.shape[0]*bdata.shape[1], *bdata.shape[2:])  # shape (len(data)//bs, bs, *) => (len(data), *)
        self.bs = bs
        self.seq_len = seq_len
        self.transform = transform
        self.PATH = PATH
        self.IMAGES_FOLDER = IMAGES_FOLDER
    
    def __len__(self):
        return len(self.bdata)
    
    def __getitem__(self, i):
        img_id = self.bdata[i][0]
        img = open_image(os.path.join(self.PATH, self.IMAGES_FOLDER, img_id + '.png'))
        y = self.bdata[i][1:].astype(np.float32)
        return self.transform(img, y)

    def batch_sampler(self, shuffle=False):
        if shuffle: # JUST FOR TESTING
            return BatchSampler(RandomSampler(self), batch_size=self.bs*self.seq_len, drop_last=True)
        return BatchSampler(SequentialSampler(self), batch_size=self.bs*self.seq_len, drop_last=True)

class BatchifiedFeedbackDataset(Dataset):
    def __init__(self, data, bs, seq_len, transform, PATH, IMAGES_FOLDER):
        bdata = batchify(data, bs)
        self.bdata = bdata.reshape(bdata.shape[0]*bdata.shape[1], *bdata.shape[2:])  # shape (len(data)//bs, bs, *) => (len(data), *)
        self.bs = bs
        self.seq_len = seq_len
        self.transform = transform
        self.PATH = PATH
        self.IMAGES_FOLDER = IMAGES_FOLDER

    def __len__(self):
        return len(self.bdata)

    def __getitem__(self, i):
        img_id = self.bdata[i][0]
        img = open_image(os.path.join(self.PATH, self.IMAGES_FOLDER, img_id + '.png'))
        y = self.bdata[i][1:].astype(np.float32)
        img, _ = self.transform(img, y) # No transformation on y supported at the moment
        y_prev = self.bdata[i-1][1:].astype(np.float32)
        return img, y_prev, y

    def batch_sampler(self):
        # Skips first element
        return BatchSampler(range(1, len(self)), batch_size=self.bs*self.seq_len, drop_last=True)

class LinearFeedbackDataset(Dataset):
    def __init__(self, df, transform, PATH, IMAGES_FOLDER):
        self.data = df.values
        self.transform = transform
        self.PATH = PATH
        self.IMAGES_FOLDER = IMAGES_FOLDER
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        img_id = self.data[i][0]
        img = open_image(os.path.join(self.PATH, self.IMAGES_FOLDER, img_id + '.png'))
        y = self.data[i][1:].astype(np.float32)
        img, _ = self.transform(img, y) # No transformation on y supported at the moment
        y_prev = self.data[i-1][1:].astype(np.float32)
        return img, y_prev, y

    def batch_sampler(self):
        # Skips first element
        return BatchSampler(range(1, len(self)), batch_size=1, drop_last=True)
