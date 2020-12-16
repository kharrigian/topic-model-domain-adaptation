
########################
### Configuration
########################

## Choose Directories to Analyze
COMPARE_DIRS = {
    "LDA":"./data/results/depression/k_latent/wolohan_clpsych/LDA/",
    "PLDA":"./data/results/depression/k_latent/wolohan_clpsych/PLDA/",
}

## Plot Directory
PLOT_DIR = "./plots/classification/sample_size/LDA/target_sample_size/"

## Fixed Variation Fields (Average Over)
MODEL_VARS = ["norm","is_average_representation"]

## Aggregation Indices
INDEX_VARS = ["alpha","beta","k_latent","k_per_label"]
COLUMN_VARS = []

## Metrics
METRIC_VARS = ["f1","precision","recall","avg_precision","auc"]

########################
### Imports
########################

## Standard Library
import os
import sys
import json
from glob import glob

## External Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

########################
### Helpers
########################

def average(x, precision=5, include_std=False):
    """

    """
    mean = np.mean(x)
    std = np.std(x)
    if include_std:
        res = "{} ({})".format(round(mean, precision), round(std,precision))
    else:
        res = "{}".format(round(mean, precision))
    return res

########################
### Load Results
########################

## Load Scores
scores_df = []
for exp_id, exp_dir in COMPARE_DIRS.items():
    exp_id_dirs = glob(f"{exp_dir}*/")
    for ed in exp_id_dirs:
        ## Load Config
        with open(f"{ed}config.json","r") as the_file:
            ed_config = json.load(the_file)
        ## Load Score Data
        if not os.path.exists(f"{ed}classification/scores.csv"):
            print(ed)
            continue
        sf_data = pd.read_csv(f"{ed}classification/scores.csv").fillna("None")
        ## Merge Metadata
        sf_data["experiment"] = exp_id
        for v in ed_config.keys():
            if v.endswith("sample_size"):
                sf_data[v] = ed_config[v]["train"]
            elif v.endswith("class_ratio"):
                sf_data[f"{v}_train"] = str(ed_config[v]["train"])
                sf_data[f"{v}_dev"] = str(ed_config[v]["dev"])
            elif v in ["C","averaging","norm","random_seed","max_iter"]:
                continue
            else:
                sf_data[v] = ed_config[v]
        ## Cache Scores
        scores_df.append(sf_data)
scores_df = pd.concat(scores_df).reset_index(drop=True)

## Aggregate Scores
metric_aggs = dict()
for metric in METRIC_VARS:
    scores_df_agg = pd.pivot_table(scores_df,
                                   index=["group"]+MODEL_VARS+INDEX_VARS,
                                   columns=["experiment","domain"]+COLUMN_VARS,
                                   values=metric,
                                   aggfunc=average).loc["development"]
    metric_aggs[metric] = scores_df_agg
metric_aggs = pd.concat(metric_aggs)

########################
### Plot Results (Heatmaps)
########################

## Establish Plot Directories
for metric in METRIC_VARS:
    metric_plot_dir = f"{PLOT_DIR}{metric}/"
    if not os.path.exists(metric_plot_dir):
        _ = os.makedirs(metric_plot_dir)

## Separate and Cache Subsets
for metric in METRIC_VARS:
    for model_ind in scores_df[MODEL_VARS].drop_duplicates().values:
        metric_agg_subset = metric_aggs.loc[metric, model_ind[0], model_ind[1]].sort_index()
        metric_agg_subset.to_csv(f"{PLOT_DIR}{metric}/{MODEL_VARS[0]}-{model_ind[0]}_{MODEL_VARS[1]}-{model_ind[1]}.csv")