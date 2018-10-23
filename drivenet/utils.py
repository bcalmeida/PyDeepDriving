# from fastai.imports import *
# from fastai.torch_imports import *

import math
from matplotlib import pyplot as plt

######################
###################### Global variables
######################

COLUMNS = [
    'img_id',
    'angle',
    'toMarking_L',
    'toMarking_M',
    'toMarking_R',
    'dist_L',
    'dist_R',
    'toMarking_LL',
    'toMarking_ML',
    'toMarking_MR',
    'toMarking_RR',
    'dist_LL',
    'dist_MM',
    'dist_RR',
    'fast'
    ]

INDIC_RANGES = [
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
    [0, 1],       # fast
]

UNIT_RANGES = [[0, 1] for _ in range(14)]

######################
###################### Plotting
######################
def plots(ims, figsize=None, rows=1, cols=None, titles=None, show_axis=False):
    if cols is None:
        cols = math.ceil(len(ims)/rows)
    else:
        rows = math.ceil(len(ims)/cols)
    if figsize is None:
        figsize = (cols*5,rows*5)
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        if not show_axis:
            sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=12)
        plt.imshow(ims[i])

# IMG_PATHS = [PATH + IMAGES_FOLDER + img_id + ".png" for img_id in df_large.img_id]
def plot_from_img_idxs(idxs, img_paths):
    imgs = [plt.imread(img_paths[i]) for i in idxs]
    titles = ["{:,}".format(i) for i in idxs]
    #plots(imgs, rows=len(imgs)//4, figsize=(20,len(imgs)), titles=titles)
    plots(imgs, cols=4, titles=titles)

######################
###################### Range transformations
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
