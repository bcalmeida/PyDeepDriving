import drive
from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np

# Fastai imports
from fastai.imports import *
from fastai.torch_imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

# Extra imports
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from fastai.dataloader import DataLoader as FastaiDataLoader

torch.cuda.set_device(0)

######################
###################### Definitions general
######################
def transform_range(x, r1, r2):
    size_r1 = r1[1] - r1[0]
    size_r2 = r2[1] - r2[0]
    len_x1 = x - r1[0]
    len_x2 = len_x1*size_r2/size_r1
    return r2[0] + len_x2

def transform_range_df(df, r1s, r2s, ignore_first=True):
    df_transformed = df.copy(deep=True)
    cols = df.columns[1:] if ignore_first else df.columns
    for i, colname in enumerate(cols):
        df_transformed[colname] = transform_range(df[colname], r1s[i], r2s[i])
    return df_transformed

def transform_range_output(o, r1s, r2s):
    o_tfm = []
    for i, v in enumerate(o):
        o_tfm.append(transform_range(v, r1s[i], r2s[i]))
    return o_tfm

######################
###################### Definitions models
######################

# def batchify(data, bs):
#     nbatch = data.shape[0] // bs
#     splits = np.split(data[:nbatch*bs], bs)
#     return np.stack(splits, axis=1)

# class BatchifiedDataset(Dataset):
#     def __init__(self, data, bs, seq_len, transform):
#         bdata = batchify(data, bs)
#         self.bdata = bdata.reshape(bdata.shape[0]*bdata.shape[1], *bdata.shape[2:])  # shape (len(data)//bs, bs, *) => (len(data), *)
#         self.bs = bs
#         self.seq_len = seq_len
#         self.transform = transform
    
#     def __len__(self):
#         return len(self.bdata)
    
#     def __getitem__(self, i):
#         img_id = self.bdata[i][0]
#         img = open_image(os.path.join(PATH, IMAGES_FOLDER, img_id + '.png'))
#         y = self.bdata[i][1:].astype(np.float32)
#         return self.transform(img, y)

#     def batch_sampler(self, shuffle=False):
#         if shuffle: # JUST FOR TESTING
#             return BatchSampler(RandomSampler(self), batch_size=self.bs*self.seq_len, drop_last=True)
#         return BatchSampler(SequentialSampler(self), batch_size=self.bs*self.seq_len, drop_last=True)

class LinearFeedbackDataset(Dataset):
    def __init__(self, df, transform):
        # bdata = batchify(data, bs)
        # self.bdata = bdata.reshape(bdata.shape[0]*bdata.shape[1], *bdata.shape[2:])  # shape (len(data)//bs, bs, *) => (len(data), *)
        # self.bs = bs
        # self.seq_len = seq_len
        self.data = df.values
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        img_id = self.data[i][0]
        img = open_image(os.path.join(PATH, IMAGES_FOLDER, img_id + '.png'))
        y = self.data[i][1:].astype(np.float32)
        img, _ = self.transform(img, y) # No transformation on y supported at the moment
        y_prev = self.data[i-1][1:].astype(np.float32)
        return img, y_prev, y

    def batch_sampler(self):
        # Skips first element
        return BatchSampler(range(1, len(self)), batch_size=1, drop_last=True)

# class CNNtoFC(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         resnet_model = resnet34(pretrained=True)
#         encoder_layers = list(resnet_model.children())[:8] + [AdaptiveConcatPool2d(), Flatten()]
#         self.encoder = nn.Sequential(*encoder_layers).cuda()
#         for param in self.encoder.parameters():
#             param.requires_grad = False
#         set_trainable(self.encoder, False) # fastai fit bug
        
#         self.linear = nn.Sequential(
#             nn.BatchNorm1d(1024),
#             nn.Dropout(p=0.25),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.BatchNorm1d(512),
#             nn.Dropout(p=0.5),
#             nn.Linear(512,14)).cuda()
#         apply_init(self.linear, kaiming_normal)
#         set_trainable(self.linear, True) # fastai fit bug
        
#     def forward(self, x):
#         encodes = self.encoder(x) # shape (seq_len*bs, 1024)
#         return self.linear(encodes) # shape (seq_len*bs, 14)

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

######################
###################### Datasets
######################

# Paths
ROOT = '/home/bcalmeida/dev/py-deep-torcs'
PATH = "data/"
IMAGES_FOLDER = "images/"
LABELS_CSV = PATH + "labels.csv"
IMAGES_BASELINE_FOLDER = "images-baseline/"
LABELS_BASELINE_CSV = PATH + "labels-baseline.csv"

df_total = pd.read_csv(LABELS_CSV)
print(df_total.head())

df_baseline = pd.read_csv(LABELS_BASELINE_CSV)
print(df_baseline.head())

# Transform output ranges
output_ranges = [
    [-0.5, 0.5],  # angle
    [-7, -2.8],   # toMarking_L (-2.8 instead 2.5)
    [-2, 3.5],    # toMarking_M
    [2.8, 7],     # toMarking_R (2.8 instead of 2.5)
    [0, 75],      # dist_L
    [0, 75],      # dist_R
    [-9.5, -4.5], # toMarking_LL (4.5 instead of 4)
    [-5.5, -0.5], # toMarking_ML
    [0.5, 5.5],   # toMarking_MR
    [4.5, 9.5],   # toMarking_RR (4.5 instead of 4)
    [0, 75],      # dist_LL
    [0, 75],      # dist_MM
    [0, 75],      # dist_RR
    [0,1],        # fast
]
desired_ranges = [[0,1] for _ in range(14)]
df_total_tfm = transform_range_df(df_total, output_ranges, desired_ranges)
print(df_total_tfm.head())

# Define training and validation datasets
tracks = [
    # [start_idx, num_lanes, track, is_alone]
    [0     , 3, 'ferris', False],
    [108942, 3, 'orange', False],
    [175695, 3, 'buildings', False],
    [240512, 2, 'desert', True],   
    [247145, 2, 'ferris', True],   
    [256593, 2, 'ferris', False],   
    [274935, 2, 'desert', False],   
    [304102, 2, 'ferris', False],   
    [327076, 2, 'grassland', True],   
    [328298, 2, 'snow', False],   
    [343748, 2, 'grassland', False],   
    [348108, 2, 'snow', False],   
    [364678, 2, 'desert', False],   
    [381132, 2, 'orange', False],   
    [409551, 2, 'ferris', False],   
    [441736, 1, 'grassland', True],   
    [443415, 1, 'desert', True],   
    [444766, 1, 'ferris', True],   
    [447804, 1, 'grassland', False],   
    [452753, 1, 'desert', False],   
    [456874, 1, 'ferris', False],   
    [463782, 1, 'grassland', True],   
    [465302, 1, 'desert', True],   
    [466657, 1, 'ferris', True],   
    [469787, 1, 'ferris', False],   
    [476438, 1, 'desert', False],   
    [480576, 1, 'grassland', False],
]

def parse_tracks(tracks):
    tracks = tracks + [[484815,0,'DUMMY',False]]
    parsed = []
    for i in range(len(tracks)-1):
        start = tracks[i][0]
        end = tracks[i+1][0]
        size = end-start
        track_name = tracks[i][2]
        num_lanes = tracks[i][1]
        is_alone = tracks[i][3]
        parsed.append([track_name, start, end, size, num_lanes, is_alone])
    return np.asarray(parsed, dtype=np.object)

def reduce_tracks(df_tracks, ratio):
    df_reduced = df_tracks.copy(deep=True)
    df_reduced.end = df_tracks.start + ((df_tracks.end - df_tracks.start)*ratio).astype(int)
    df_reduced.length = df_reduced.end - df_reduced.start
    return df_reduced

def print_dataset_tracks_stats(df_tracks, val_tracks_mask):
    total_size =  df_tracks.length.sum()
    trn_size = df_tracks[~val_tracks_mask].length.sum()
    val_size = df_tracks[val_tracks_mask].length.sum()
    val_ratio = val_size / total_size
    
    trn_3lane = df_tracks[~val_tracks_mask & (df_tracks.num_lanes==3)].length.sum()
    trn_2lane = df_tracks[~val_tracks_mask & (df_tracks.num_lanes==2)].length.sum()
    trn_1lane = df_tracks[~val_tracks_mask & (df_tracks.num_lanes==1)].length.sum()
    trn_lanes = [trn_3lane, trn_2lane, trn_1lane]
    trn_lane_ratios = [lane/trn_size for lane in trn_lanes]
    
    val_3lane = df_tracks[val_tracks_mask & (df_tracks.num_lanes==3)].length.sum()
    val_2lane = df_tracks[val_tracks_mask & (df_tracks.num_lanes==2)].length.sum()
    val_1lane = df_tracks[val_tracks_mask & (df_tracks.num_lanes==1)].length.sum()
    val_lanes = [val_3lane, val_2lane, val_1lane]
    val_lane_ratios = [lane/val_size for lane in val_lanes]
    
    print("total_size:", total_size)
    print("trn_size:", trn_size)
    print("val_size:", val_size)
    print("val_ratio: %.2f" % val_ratio)
    print()
    print("lanes statistics (3/2/1 lanes):")
    print("trn_lanes:", tuple(trn_lanes))
    print("trn_lane_ratios: %.2f, %.2f, %.2f" % tuple(trn_lane_ratios))
    print("val_lanes:", tuple(val_lanes))
    print("val_lane_ratios: %.2f, %.2f, %.2f" % tuple(val_lane_ratios))

def df_tracks_to_idxs(df_tracks, val_tracks_mask):
    trn_ranges = df_tracks[~val_tracks_mask][['start', 'end']].values
    trn_idxs = [list(range(start, end)) for start, end in trn_ranges]
    trn_idxs = [x for z in trn_idxs for x in z] # flatten
    
    val_ranges = df_tracks[val_tracks_mask][['start', 'end']].values
    val_idxs = [list(range(start, end)) for start, end in val_ranges]
    val_idxs = [x for z in val_idxs for x in z] # flatten
    return trn_idxs, val_idxs

df_tracks = pd.DataFrame(data=parse_tracks(tracks), columns=['track_name', 'start', 'end', 'length', 'num_lanes', 'is_alone'])
val_tracks_mask = (df_tracks.track_name == 'buildings') | (df_tracks.track_name == 'grassland')

print("=== TOTAL DATASET ===")
print_dataset_tracks_stats(df_tracks, val_tracks_mask)
print()

trn_idxs, val_idxs = df_tracks_to_idxs(df_tracks, val_tracks_mask)
df_total_trn, df_total_val = df_total_tfm.iloc[trn_idxs], df_total_tfm.iloc[val_idxs]
print("df_total_trn, df_total_val", len(df_total_trn), len(df_total_val))
print()

print("=== LARGE DATASET ===")
df_tracks_large = pd.concat([
    # Force 60/30/10 3lane/2lane/1lane distribution, while losing the minimum amount of data
    # Also multiply trn_set by 0.7 to keep val/trn ratio not so low
    reduce_tracks(df_tracks[~val_tracks_mask & (df_tracks.num_lanes == 3)], (60/44)*(44/60)*0.7),
    reduce_tracks(df_tracks[~val_tracks_mask & (df_tracks.num_lanes == 2)], (30/49)*(44/60)*0.7),
    reduce_tracks(df_tracks[~val_tracks_mask & (df_tracks.num_lanes == 1)], (10/ 8)*(44/60)*0.7),
    
    reduce_tracks(df_tracks[val_tracks_mask & (df_tracks.num_lanes == 3)], (60/78)*(7/30)),
    reduce_tracks(df_tracks[val_tracks_mask & (df_tracks.num_lanes == 2)], (30/ 7)*(7/30)),
    reduce_tracks(df_tracks[val_tracks_mask & (df_tracks.num_lanes == 1)], (10/15)*(7/30)),
]).sort_index()
print_dataset_tracks_stats(df_tracks_large, val_tracks_mask)
print()

trn_idxs, val_idxs = df_tracks_to_idxs(df_tracks_large, val_tracks_mask)
df_large_trn, df_large_val = df_total_tfm.iloc[trn_idxs], df_total_tfm.iloc[val_idxs]
print("df_large_trn, df_large_val", len(df_large_trn), len(df_large_val))
print()

print("=== MEDIUM DATASET ===")
df_tracks_medium = pd.concat([
    reduce_tracks(df_tracks_large, 0.5),
]).sort_index()
print_dataset_tracks_stats(df_tracks_medium, val_tracks_mask)
print()

trn_idxs, val_idxs = df_tracks_to_idxs(df_tracks_medium, val_tracks_mask)
df_medium_trn, df_medium_val = df_total_tfm.iloc[trn_idxs], df_total_tfm.iloc[val_idxs]
print("df_medium_trn, df_medium_val", len(df_medium_trn), len(df_medium_val))
print()

print("=== SMALL DATASET ===")
df_tracks_small = pd.concat([
    reduce_tracks(df_tracks_large, 0.1),
]).sort_index()
print_dataset_tracks_stats(df_tracks_small, val_tracks_mask)
print()

trn_idxs, val_idxs = df_tracks_to_idxs(df_tracks_small, val_tracks_mask)
df_small_trn, df_small_val = df_total_tfm.iloc[trn_idxs], df_total_tfm.iloc[val_idxs]
print("df_small_trn, df_small_val", len(df_small_trn), len(df_small_val))
print()

######################
###################### Network setup
######################
# More settings
arch = resnet34
sz=210
tfms = tfms_from_model(arch, sz, crop_type=CropType.NO)
trn_tfms, val_tfms = tfms

# Training settings
bs = 1
seq_len = 1

# Datasets and Dataloaders
trn_lds = LinearFeedbackDataset(df_small_trn, trn_tfms)
val_lds = LinearFeedbackDataset(df_small_val, trn_tfms)
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
model.eval()

# opt_fn is used like this: optimizer = opt_fn(trainable_params(model), lr=1e-1)
opt_fn = partial(optim.SGD, momentum=0.9)
criterion = F.l1_loss

learner = Learner(
    MockedData(trn_dl, val_dl),
    CustomModel(model, layer_groups),
    metrics=metrics,
    opt_fn=opt_fn,
    crit=criterion,
    tmp_name=os.path.join(ROOT, PATH, 'tmp'),
    models_name=os.path.join(ROOT, PATH, 'models', model_folder),
)
# clip and reg_fn needs shouldn't be passed to the constructor because it sets as None anyway...
# learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learner.clip=0.4

############
############ Model loading
############

learner.load('nb23-sgd-bs14-7frz-21unfrz')

######################
###################### Network test
######################
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

metrics = [
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
]

# added ipynb 22
def acc_fast(preds, y):
    idx = 13
    preds = (preds[:, idx] > 0.5)
    y = (y[:, idx] > 0.5)
    return (preds == y).float().mean()

metrics.append(acc_fast)

### Network testing methods
# diff ipynb 22
def predict_from_df_raw(model, df):
    """
    returns df_preds_tfm, y_tfm, df_preds, df_y
    """
    df_tfm = transform_range_df(df, output_ranges, desired_ranges)
    return predict_from_df_tfm(model, df_tfm)

# added ipynb 22
def predict_from_df_tfm(model, df_tfm):
    """
    returns df_preds_tfm, y_tfm, df_preds, df_y
    """
    data = ImageRegressorData.from_data_frames(PATH, IMAGES_FOLDER, df_tfm, df_tfm,
                                               suffix='.png', bs=64, tfms=tfms)
    preds_tfm, y_tfm = predict_with_targs(model, data.val_dl)
    
    df_preds_tfm = pd.DataFrame(data=preds_tfm, columns=df_tfm.columns[1:], copy=True)
    df_y_tfm = pd.DataFrame(data=y_tfm, columns=df_tfm.columns[1:], copy=True)

    df_preds = transform_range_df(df_preds_tfm, desired_ranges, output_ranges, ignore_first=False)
    df_y = transform_range_df(df_y_tfm, desired_ranges, output_ranges, ignore_first=False)
    
    return df_preds_tfm, df_y_tfm, df_preds, df_y

def calc_all_metrics(preds, y):
    preds = T(preds)
    y = T(y)
    res = []
    for f in metrics:
        res.append(f(preds, y))
    return res

def print_calced_metrics(*mets):
    header = ""
    lines = []
    for f in metrics:
        header += "%-11s" % f.__name__
    for met in mets:
        line = ""
        for v in met:
            line += "%-11.3f" % v
        lines.append(line)
    
    print(header)
    for line in lines:
        print(line)

# diff ipynb 22
def print_calced_metrics_from_df_raw(model, df):
    res = predict_from_df_raw(model, df)
    print_calced_metrics(calc_all_metrics(res[0], res[1]), calc_all_metrics(res[2], res[3]))

# added ipynb 22
def print_calced_metrics_from_df_tfm(model, df_tfm):
    res = predict_from_df_tfm(model, df_tfm)
    print_calced_metrics(calc_all_metrics(res[0], res[1]), calc_all_metrics(res[2], res[3]))

# added ipynb 22
def predict_from_dl(model, dl):
    preds, y = predict_with_targs(model, dl)
    
    df_preds_tfm = pd.DataFrame(data=preds, columns=df_total.columns[1:], copy=True)
    df_y_tfm = pd.DataFrame(data=y, columns=df_total.columns[1:], copy=True)
    
    df_preds_raw = transform_range_df(df_preds_tfm, desired_ranges, output_ranges, ignore_first=False)
    df_y_raw = transform_range_df(df_y_tfm, desired_ranges, output_ranges, ignore_first=False)

    return df_preds_tfm, df_y_tfm, df_preds_raw, df_y_raw

# added ipynb 22
def print_calced_metrics_from_dl(model, dl):
    res = predict_from_dl(model, dl)
    print_calced_metrics(calc_all_metrics(res[0], res[1]), calc_all_metrics(res[2], res[3]))


# print_calced_metrics_from_dl(learner.model, val_dl)


######################
###################### Test driving
######################

# import pdb; pdb.set_trace()

def format_indicators(pred_indicators):
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
    indicators_formatted = [0] * 16
    indicators_formatted[0]  = pred_indicators[13] # fast
    indicators_formatted[1]  = pred_indicators[4]  # dist_L
    indicators_formatted[2]  = pred_indicators[5]  # dist_R
    indicators_formatted[3]  = pred_indicators[1]  # toMarking_L
    indicators_formatted[4]  = pred_indicators[2]  # toMarking_M
    indicators_formatted[5]  = pred_indicators[3]  # toMarking_R
    indicators_formatted[6]  = pred_indicators[10] # dist_LL
    indicators_formatted[7]  = pred_indicators[11] # dist_MM
    indicators_formatted[8]  = pred_indicators[12] # dist_RR
    indicators_formatted[9]  = pred_indicators[6]  # toMarking_LL
    indicators_formatted[10] = pred_indicators[7]  # toMarking_ML
    indicators_formatted[11] = pred_indicators[8]  # toMarking_MR
    indicators_formatted[12] = pred_indicators[9]  # toMarking_RR
    indicators_formatted[13] = 0
    indicators_formatted[14] = pred_indicators[0] # angle
    indicators_formatted[15] = 0
    return indicators_formatted

@contextmanager
def context(*args, **kwds):
    drive.setup_shared_memory()
    drive.setup_opencv()
    try:
        yield None
    finally:
        drive.close_shared_memory()
        drive.close_opencv()

learner.model.eval()

WIDTH = 280
HEIGHT = 210
with context() as _:
    drive.pause(False) # TORCS may share images and ground truth
    print("Controlling: ", drive.is_controlling())
    drive.set_control(True)
    print("Controlling: ", drive.is_controlling())
    input("Press key to start...")
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
            pred_indicators = transform_range_output(to_np(output[0]), desired_ranges, output_ranges)
            print("network raw output", output)
            print("pred_indicators", pred_indicators)

            indicators_formatted = format_indicators(pred_indicators)
            print("indicators_formatted", indicators_formatted)

            ground_truth = drive.read_indicators()
            print("ground_truth", ground_truth)

            drive.controller(indicators_formatted)
            drive.update_visualizations(indicators_formatted, ground_truth)
            drive.write(False) # Shared data read, and TORCS may continue
            drive.wait_key(1)
