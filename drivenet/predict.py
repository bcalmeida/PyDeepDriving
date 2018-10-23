# from fastai.imports import *
# from fastai.torch_imports import *
# from fastai.transforms import *
# from fastai.conv_learner import *
# from fastai.model import *
# from fastai.dataset import *
# from fastai.sgdr import *
# from fastai.plots import *

import pandas as pd
from fastai.model import predict_with_targs
from .utils import transform_range_df, INDIC_RANGES, UNIT_RANGES, COLUMNS
from .metrics import print_calced_metrics, calc_all_metrics

# def predict_from_df_raw(model, df):
#     """
#     returns df_preds_tfm, y_tfm, df_preds, df_y
#     """
#     df_tfm = transform_range_df(df, INDIC_RANGES, UNIT_RANGES)
#     return predict_from_df_tfm(model, df_tfm)

# # added ipynb 22
# def predict_from_df_tfm(model, df_tfm):
#     """
#     returns df_preds_tfm, y_tfm, df_preds, df_y
#     """
#     data = ImageRegressorData.from_data_frames(PATH, IMAGES_FOLDER, df_tfm, df_tfm,
#                                                suffix='.png', bs=64, tfms=tfms)
#     preds_tfm, y_tfm = predict_with_targs(model, data.val_dl)
#    
#     df_preds_tfm = pd.DataFrame(data=preds_tfm, columns=df_tfm.columns[1:], copy=True)
#     df_y_tfm = pd.DataFrame(data=y_tfm, columns=df_tfm.columns[1:], copy=True)
#
#     df_preds = transform_range_df(df_preds_tfm, UNIT_RANGES, INDIC_RANGES, ignore_first=False)
#     df_y = transform_range_df(df_y_tfm, UNIT_RANGES, INDIC_RANGES, ignore_first=False)
#    
#     return df_preds_tfm, df_y_tfm, df_preds, df_y

# added ipynb 22
def predict_from_dl(model, dl):
    preds, y = predict_with_targs(model, dl)
    
    df_preds_tfm = pd.DataFrame(data=preds, columns=COLUMNS[1:], copy=True)
    df_y_tfm = pd.DataFrame(data=y, columns=COLUMNS[1:], copy=True)
    
    df_preds_raw = transform_range_df(df_preds_tfm, UNIT_RANGES, INDIC_RANGES, ignore_first=False)
    df_y_raw = transform_range_df(df_y_tfm, UNIT_RANGES, INDIC_RANGES, ignore_first=False)

    return df_preds_tfm, df_y_tfm, df_preds_raw, df_y_raw

# # diff ipynb 22
# def print_calced_metrics_from_df_raw(model, df):
#     res = predict_from_df_raw(model, df)
#     print_calced_metrics(calc_all_metrics(res[0], res[1]), calc_all_metrics(res[2], res[3]))

# # added ipynb 22
# def print_calced_metrics_from_df_tfm(model, df_tfm):
#     res = predict_from_df_tfm(model, df_tfm)
#     print_calced_metrics(calc_all_metrics(res[0], res[1]), calc_all_metrics(res[2], res[3]))

# added ipynb 22
def print_calced_metrics_from_dl(model, dl):
    res = predict_from_dl(model, dl)
    print_calced_metrics(calc_all_metrics(res[0], res[1]), calc_all_metrics(res[2], res[3]))