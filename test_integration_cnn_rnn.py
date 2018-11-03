import os
from contextlib import contextmanager
from functools import partial
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.models import resnet34

import drive

from drivenet.data_handlers import LinearFeedbackDataset
from drivenet.cnn_rnn_net import CNNtoRNNFeedback
from drivenet.ds_creation import setup_dataset_dfs
from drivenet.fastaiwrap import CustomModel, MockedData
from drivenet.metrics import METRICS, calc_all_metrics, print_calced_metrics
from drivenet.predict import print_calced_metrics_from_dl
from drivenet.utils import transform_range_output, INDIC_RANGES, UNIT_RANGES

from fastai.transforms import tfms_from_model, CropType
from fastai.dataloader import DataLoader as FastaiDataLoader
from fastai.learner import Learner
from fastai.core import T, to_np

torch.cuda.set_device(0)

# Paths
ROOT = '/home/bcalmeida/dev/py-deep-torcs'
PATH = "data/"
IMAGES_FOLDER = "images/"
LABELS_CSV = PATH + "labels.csv"
IMAGES_BASELINE_FOLDER = "images-baseline/"
LABELS_BASELINE_CSV = PATH + "labels-baseline.csv"

(df_baseline,
df_total,
df_total_tfm,
df_total_trn,
df_total_val,
df_large_trn,
df_large_val,
df_medium_trn,
df_medium_val,
df_small_trn,
df_small_val) = setup_dataset_dfs(LABELS_CSV, LABELS_BASELINE_CSV)

######################
###################### Network setup
######################
# More settings
arch = resnet34
sz=210
trn_tfms, _ = tfms_from_model(arch, sz, crop_type=CropType.NO)
trn_tfms.tfms.pop(1) # Remove cropping to keep it rectangular

# Training settings
bs = 1
seq_len = 1

# Datasets and Dataloaders
trn_lds = LinearFeedbackDataset(df_small_trn, trn_tfms, PATH, IMAGES_FOLDER)
val_lds = LinearFeedbackDataset(df_small_val, trn_tfms, PATH, IMAGES_FOLDER)
trn_dl = FastaiDataLoader(trn_lds, batch_sampler=trn_lds.batch_sampler())
val_dl = FastaiDataLoader(val_lds, batch_sampler=val_lds.batch_sampler())

# Model
model_folder = "CNNtoRNNFeedback"
model = CNNtoRNNFeedback(1024, 200, 2, seq_len, bs, 14, use_ground_truth=False)
layer_groups = [
    list(model.encoder.children())[:6],
    list(model.encoder.children())[6:],
    [model.lstm, model.linear],
]

# opt_fn is used like this: optimizer = opt_fn(trainable_params(model), lr=1e-1)
opt_fn = partial(optim.SGD, momentum=0.9)
criterion = F.l1_loss

learner = Learner(
    MockedData(trn_dl, val_dl),
    CustomModel(model, layer_groups),
    metrics=METRICS,
    opt_fn=opt_fn,
    crit=criterion,
    tmp_name=os.path.join(ROOT, PATH, 'tmp'),
    models_name=os.path.join(ROOT, PATH, 'models', model_folder),
)
# clip and reg_fn needs shouldn't be passed to the constructor because it sets as None anyway...
# learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learner.clip = 0.4


############
############ Model loading
############

# learner.load('nb25-sz210-pre-B')
learner.load('nb26-sz210-c')
learner.model.eval()

# print_calced_metrics_from_dl(learner.model, val_dl)

######################
###################### Test driving
######################

# import pdb; pdb.set_trace()

"""
Indicators are formatted differently on the controller
Cpp controller:
indicators[0]  = shared->fast;
indicators[1]  = shared->dist_L;
indicators[2]  = shared->dist_R;
indicators[3]  = shared->toMarking_L;
indicators[4]  = shared->toMarking_M;
indicators[5]  = shared->toMarking_R;
indicators[6]  = shared->dist_LL;
indicators[7]  = shared->dist_MM;
indicators[8]  = shared->dist_RR;
indicators[9]  = shared->toMarking_LL;
indicators[10] = shared->toMarking_ML;
indicators[11] = shared->toMarking_MR;
indicators[12] = shared->toMarking_RR;
indicators[13] = shared->toMiddle;
indicators[14] = shared->angle;
indicators[15] = shared->speed;

Python network:
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
"""
INDS_CNTRL_NET_MAPPINGS = [
    [0, 13],
    [1, 4],
    [2, 5],
    [3, 1],
    [4, 2],
    [5, 3],
    [6, 10],
    [7, 11],
    [8, 12],
    [9, 6],
    [10, 7],
    [11, 8],
    [12, 9],
    [14, 0],
]

def inds_net_to_cntrl(inds_net):
    inds_cntrl = [0] * 16
    for a,b in INDS_CNTRL_NET_MAPPINGS:
        inds_cntrl[a] = inds_net[b]
    return inds_cntrl

def inds_ctrl_to_net(inds_cntrl):
    inds_net = [0] * 14
    for a,b in INDS_CNTRL_NET_MAPPINGS:
        inds_net[b] = inds_cntrl[a]
    return inds_net

@contextmanager
def context(*args, **kwds):
    drive.setup_shared_memory()
    drive.setup_opencv()
    try:
        yield None
    finally:
        drive.close_shared_memory()
        drive.close_opencv()

WIDTH = 280
HEIGHT = 210
with context() as _:
    drive.pause(False) # TORCS may share images and ground truth
    print("Controlling: ", drive.is_controlling())
    drive.set_control(True)
    print("Controlling: ", drive.is_controlling())
    input("Press key to start...")

    ### metrics
    i = 0
    total_mets_unit = np.zeros(len(METRICS))
    total_mets_indic = np.zeros(len(METRICS))
    # total_mae_unit = 0
    # total_mae_indic = 0
    ### metrics

    while True:
        if drive.is_written():
            print("Reading img")
            img = drive.read_image() # HWC, BGR
            img_np = np.asarray(img).reshape(HEIGHT, WIDTH, 3)[:,:,::-1] #.astype('uint8') # HWC, RGB

            # import pdb; pdb.set_trace()
            #plt.imshow(img_np)
            #plt.show()

            img_np = (img_np/255).astype('float32')
            x = trn_tfms(img_np)[np.newaxis, ...] # shape (210, 280, 3) -> (3, 210, 210) -> (1, 3, 210, 210)
            x = Variable(T(x),requires_grad=False, volatile=True)
            output = learner.model(x, None) # shape (1, 14)
            pred_indicators = transform_range_output(to_np(output[0]), UNIT_RANGES, INDIC_RANGES)
            print("network raw output", output)
            print("pred_indicators", pred_indicators)

            indicators_formatted = inds_net_to_cntrl(pred_indicators)
            print("indicators_formatted", indicators_formatted)

            ground_truth = drive.read_indicators()
            print("ground_truth", ground_truth)

            drive.controller(indicators_formatted)
            drive.update_visualizations(indicators_formatted, ground_truth)
            
            ########################################################
            ############################ metrics
            ########################################################
            inds_netfmt_unitrng = to_np(output[0])
            inds_netfmt_indicrng = transform_range_output(inds_netfmt_unitrng, UNIT_RANGES, INDIC_RANGES)

            gtruth_netfmt_indicrng = inds_ctrl_to_net(ground_truth)
            gtruth_netfmt_unitrng = transform_range_output(gtruth_netfmt_indicrng, INDIC_RANGES, UNIT_RANGES)

            mets_unit = calc_all_metrics([inds_netfmt_unitrng], [gtruth_netfmt_unitrng])
            mets_indic = calc_all_metrics([inds_netfmt_indicrng], [gtruth_netfmt_indicrng])

            total_mets_unit += mets_unit
            total_mets_indic += mets_indic

            i += 1
            avg_mets_unit = total_mets_unit / i
            avg_mets_indic = total_mets_indic / i

            print("frame mae: unit: %.4f, indic: %.4f" % (mets_unit[1], mets_indic[1]))
            print("avg mae: unit: %.4f, indic: %.4f" % (avg_mets_unit[1], avg_mets_indic[1]))

            ########################################################
            ########################################################
            ########################################################

            drive.write(False) # Shared data read, and TORCS may continue
            drive.wait_key(1)
