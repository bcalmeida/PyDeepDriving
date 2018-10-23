# from fastai.imports import *
# from fastai.torch_imports import *
# from fastai.transforms import *
# from fastai.conv_learner import *
# from fastai.model import *
# from fastai.dataset import *
# from fastai.sgdr import *
# from fastai.plots import *

import os
import numpy as np
from fastai.core import BasicModel
from fastai.dataset import ImageData, FilesIndexArrayRegressionDataset, split_by_idx

class MockedData(object):
    def __init__(self, trn_dl, val_dl):
        self.trn_dl = trn_dl
        self.val_dl = val_dl
        # No need for path if you pass tmp_name and models_name as absolute paths to learner constructor
        # self.path = PATH # It will save tmp files in PATH/tmp

class CustomModel(BasicModel):
    """
    class BasicModel():
        def __init__(self,model,name='unnamed'): self.model,self.name = model,name
        def get_layer_groups(self, do_fc=False): return children(self.model)
    
    class SingleModel(BasicModel):
        def get_layer_groups(self): return [self.model]
    """
    def __init__(self, model, layer_groups):
        super().__init__(model)
        self.layer_groups = layer_groups
    
    def get_layer_groups(self):
        return self.layer_groups

# inspirations: ColumnarModelData/ColumnarDataset and ImageClassiferData
class ImageRegressorData(ImageData):
    @classmethod
    def from_data_frame(cls, path, folder, df, val_idxs,
                        suffix='', bs=64, tfms=(None,None), test_df=None, num_workers=8):
        [(val_df, trn_df)] = split_by_idx(val_idxs, df)
        return cls.from_data_frames(path, folder, trn_df, val_df,
                                suffix=suffix, bs=bs, tfms=tfms, test_df=test_df, num_workers=num_workers)
    
    @classmethod
    def from_data_frames(cls, path, folder, trn_df, val_df,
                         suffix='', bs=64, tfms=(None,None), test_df=None, num_workers=8):

        trn_img_ids = trn_df.iloc[:, 0] # First column
        val_img_ids = val_df.iloc[:, 0]

        trn_fnames = [os.path.join(folder, str(fn)+suffix) for fn in trn_img_ids]
        val_fnames = [os.path.join(folder, str(fn)+suffix) for fn in val_img_ids]
        
        test_fnames = None
        if test_df:
            test_img_ids = test_df.iloc[:, 0]
            test_fnames = [os.path.join(folder, str(fn)+suffix) for fn in test_img_ids]
        
        trn_y = trn_df.iloc[:,1:].values.astype(np.float32) # All other columns
        val_y = val_df.iloc[:,1:].values.astype(np.float32)

        y_names = trn_df.columns[1:].values
        
        datasets = cls.get_ds(FilesIndexArrayRegressionDataset, (trn_fnames, trn_y), (val_fnames, val_y), tfms,
                              path=path, test=test_fnames)
        return cls(path, datasets, bs, num_workers, classes=y_names)