# from fastai.imports import *
# from fastai.torch_imports import *

import numpy as np
import pandas as pd
from .utils import transform_range_df, INDIC_RANGES, UNIT_RANGES

# Define training and validation datasets
TRACKS = [
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

def setup_dataset_dfs(LABELS_CSV, LABELS_BASELINE_CSV):
    df_total = pd.read_csv(LABELS_CSV)
    print(df_total.head())

    df_baseline = pd.read_csv(LABELS_BASELINE_CSV)
    print(df_baseline.head())

    df_total_tfm = transform_range_df(df_total, INDIC_RANGES, UNIT_RANGES)
    print(df_total_tfm.head())

    df_tracks = pd.DataFrame(data=parse_tracks(TRACKS), columns=['track_name', 'start', 'end', 'length', 'num_lanes', 'is_alone'])
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

    return (df_baseline,
            df_total,
            df_total_tfm,
            df_total_trn,
            df_total_val,
            df_large_trn,
            df_large_val,
            df_medium_trn,
            df_medium_val,
            df_small_trn,
            df_small_val)