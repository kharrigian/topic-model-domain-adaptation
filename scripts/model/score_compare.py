
"""
Direct comparison of performance from multiple models
"""

## Script Flags
CROSS_VALIDATION = True
EVAL_SET = "test"
EVAL_DOMAIN = "target"

## Filters
FILTERS = {
}

## Where Are Results Stored
RESULT_DIR = "./data/results/depression/"

## Comparisons
COMPARISONS = [
    ## CLPsych -> Multitask
    (
        f"{RESULT_DIR}optimized/LDA/source-clpsych_deduped_target-multitask_use_plda-False_C-1_k_latent-150_norm-l2_alpha-0.01_beta-0.01/",
        f"{RESULT_DIR}optimized/PLDA/source-clpsych_deduped_target-multitask_use_plda-True_C-5_k_latent-75_k_per_label-25_norm-l2_alpha-0.01_beta-0.01/"
    ),
    ## CLPsych -> SMHD
    (
        f"{RESULT_DIR}optimized/LDA/source-clpsych_deduped_target-smhd_use_plda-False_C-1_k_latent-50_norm-l2_alpha-0.01_beta-0.01/",
        f"{RESULT_DIR}optimized/PLDA/source-clpsych_deduped_target-smhd_use_plda-True_C-10_k_latent-75_k_per_label-50_norm-l2_alpha-0.01_beta-0.01/"
    ),
    ## CLPsych -> Wolohan
    (
        f"{RESULT_DIR}optimized/LDA/source-clpsych_deduped_target-wolohan_use_plda-False_C-0.3_k_latent-50_norm-l2_alpha-0.01_beta-0.01/",
        f"{RESULT_DIR}optimized/PLDA/source-clpsych_deduped_target-wolohan_use_plda-True_C-0.001_k_latent-25_k_per_label-50_norm-l2_alpha-0.01_beta-0.01/"
    ),
    ## Multitask -> CLPsych
    (
        f"{RESULT_DIR}optimized/LDA/source-multitask_target-clpsych_deduped_use_plda-False_C-10_k_latent-200_norm-l2_alpha-0.01_beta-0.01/",
        f"{RESULT_DIR}optimized/PLDA/source-multitask_target-clpsych_deduped_use_plda-True_C-100_k_latent-50_k_per_label-75_norm-l2_alpha-0.01_beta-0.01/"
    ),
    ## Multitask -> Wolohan
    (
        f"{RESULT_DIR}optimized/LDA/source-multitask_target-wolohan_use_plda-False_C-50_k_latent-100_norm-l2_alpha-0.01_beta-0.01/",
        f"{RESULT_DIR}optimized/PLDA/source-multitask_target-wolohan_use_plda-True_C-50_k_latent-75_k_per_label-75_norm-l2_alpha-0.01_beta-0.01/"
    ),
    ## Multitask -> SMHD
    (
        f"{RESULT_DIR}optimized/LDA/source-multitask_target-smhd_use_plda-False_C-5_k_latent-50_norm-l2_alpha-0.01_beta-0.01/",
        f"{RESULT_DIR}optimized/PLDA/source-multitask_target-smhd_use_plda-True_C-10_k_latent-200_k_per_label-25_norm-l2_alpha-0.01_beta-0.01/"
    ),
    ## SMHD -> CLPsych
    (
        f"{RESULT_DIR}optimized/LDA/source-smhd_target-clpsych_deduped_use_plda-False_C-100_k_latent-75_norm-l2_alpha-0.01_beta-0.01/",
        f"{RESULT_DIR}optimized/PLDA/source-smhd_target-clpsych_deduped_use_plda-True_C-50_k_latent-75_k_per_label-100_norm-l2_alpha-0.01_beta-0.01/"
    ),
    ## SMHD -> Multitask
    (
        f"{RESULT_DIR}optimized/LDA/source-smhd_target-multitask_use_plda-False_C-0.3_k_latent-25_norm-l2_alpha-0.01_beta-0.01/",
        f"{RESULT_DIR}optimized/PLDA/source-smhd_target-multitask_use_plda-True_C-10_k_latent-200_k_per_label-100_norm-l2_alpha-0.01_beta-0.01/"
    ),
    ## SMHD -> Wolohan
    (
        f"{RESULT_DIR}optimized/LDA/source-smhd_target-wolohan_use_plda-False_C-50_k_latent-150_norm-l2_alpha-0.01_beta-0.01/",
        f"{RESULT_DIR}optimized/PLDA/source-smhd_target-wolohan_use_plda-True_C-100_k_latent-75_k_per_label-25_norm-l2_alpha-0.01_beta-0.01/"
    ),
    ## Wolohan -> CLPsych
    (
        f"{RESULT_DIR}optimized/LDA/source-wolohan_target-clpsych_deduped_use_plda-False_C-100_k_latent-75_norm-l2_alpha-0.01_beta-0.01/",
        f"{RESULT_DIR}optimized/PLDA/source-wolohan_target-clpsych_deduped_use_plda-True_C-0.1_k_latent-100_k_per_label-50_norm-l2_alpha-0.01_beta-0.01/"
    ),
    ## Wolohan -> Multitask
    (
        f"{RESULT_DIR}optimized/LDA/source-wolohan_target-multitask_use_plda-False_C-0.1_k_latent-200_norm-l2_alpha-0.01_beta-0.01/",
        f"{RESULT_DIR}optimized/PLDA/source-wolohan_target-multitask_use_plda-True_C-5_k_latent-100_k_per_label-50_norm-l2_alpha-0.01_beta-0.01/"
    ),
    ## Wolohan -> SMHD
    (
        f"{RESULT_DIR}optimized/LDA/source-wolohan_target-smhd_use_plda-False_C-0.001_k_latent-200_norm-l2_alpha-0.01_beta-0.01/",
        f"{RESULT_DIR}optimized/PLDA/source-wolohan_target-smhd_use_plda-True_C-0.001_k_latent-150_k_per_label-75_norm-l2_alpha-0.01_beta-0.01/"
    ),
]


###################
### Imports
###################

## Standard Library
import os
import sys
import json
from glob import glob

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_rel

## Local
from mhlib.util.helpers import flatten
from mhlib.util.logging import initialize_logger

###################
### Globals
###################

## Logging
LOGGER = initialize_logger()

## Metrics
ALL_METRICS = ["auc","avg_precision","f1","precision","recall"]

## Logging Order
DATASET_ORDER = ["clpsych_deduped","multitask","smhd","wolohan"]

###################
### Globals
###################

def load_scores(file,
                config):
    """

    """
    ## Load Data
    scores = pd.read_csv(file).fillna("None")
    ## Add Metadata
    for v in config.keys():
        if v.endswith("sample_size"):
            for group in ["train","dev","test"]:
                scores[f"{group}_{v}"] = config[v].get(group,None)
        elif v.endswith("class_ratio"):
            scores[f"{v}_train"] = str(config[v]["train"])
            scores[f"{v}_dev"] = str(config[v]["dev"])
        elif v == "topic_model_data":
            for ds in ["source","target"]:
                scores[f"{v}_{ds}"] = config[v][ds]
        elif v in ["C","averaging","norm","random_seed","max_iter"]:
            continue
        else:
            scores[v] = config[v]
    ## Filtering
    for fkey, fvals in FILTERS.items():
        fval_set = set(fvals)
        scores = scores.loc[scores[fkey].isin(fval_set)]
    if len(scores) == 0:
        return None
    else:
        scores = scores.reset_index(drop=True).copy()
    return scores

def summarize_compare(values, threshold=0.05):
    """

    """
    ## Format
    item_a = np.array(values.iloc[0][0])
    item_b = np.array(values.iloc[0][1])
    ## Score
    mu = np.mean(item_b - item_a)
    std = np.std(item_b - item_a)
    t, p = ttest_rel(item_a, item_b)
    ## Formatting
    result = "{:.3f} ({:.3f}){}".format(mu, std, "*" if p < threshold else "")
    return result

##################
### Analysis
##################

## Cache
scores_df = []

## Cycle Through Comparisons
for i, (a, b) in enumerate(COMPARISONS):
    for d, di in zip([a, b],["model_a","model_b"]):
        ## Load Config
        with open(f"{d}config.json","r") as the_file:
            config = json.load(the_file)
        ## Distinguish
        if CROSS_VALIDATION:
            d_score_files = sorted(glob(f"{d}fold-*/classification/scores.csv"))
        else:
            d_score_files = [f"{d}classification/scores.csv"]
        ## Load Scores
        for file in d_score_files:
            file_scores = load_scores(file, config)
            if file_scores is None:
                LOGGER.warning(f"No data: {file}")
                continue
            if not CROSS_VALIDATION:
                file_scores["fold"] = 1
            file_scores["comparison_id"] = i
            file_scores["comparison_model"] = di
            scores_df.append(file_scores)

## Concatenate
scores_df = pd.concat(scores_df).reset_index(drop=True)

## Isolate Evaluation Set
scores_df = scores_df.loc[(scores_df["group"]==EVAL_SET)&
                          (scores_df["domain"]==EVAL_DOMAIN)].reset_index(drop=True).copy()

## Computations
model_comparisons = []
for cid in scores_df["comparison_id"].unique():
    ## Cache
    cid_data = {"comparison_id":cid}
    ## Scores
    cid_scores_df = scores_df.set_index("comparison_id").loc[cid]
    cid_a = cid_scores_df.set_index("comparison_model").loc["model_a"].sort_values("fold")
    cid_b = cid_scores_df.set_index("comparison_model").loc["model_b"].sort_values("fold")
    ## Align Folds
    cid_folds = sorted(set(cid_a["fold"]) & set(cid_b["fold"]))
    if len(cid_folds) == 0:
        continue
    cid_a = cid_a.loc[cid_a["fold"].isin(cid_folds)]
    cid_b = cid_b.loc[cid_b["fold"].isin(cid_folds)]
    ## Check Data
    if len(cid_a) != len(cid_b):
        continue
    failed = False
    for domain in ["source","target"]:
        if len(set(cid_scores_df[domain])) != 1:
            failed = True
            continue
        cid_data[domain] = cid_scores_df[domain].unique()[0]    
    ## Get Metric Values
    for metric in ALL_METRICS:
        cid_data[metric] = [cid_a[metric].tolist(), cid_b[metric].tolist()]
    model_comparisons.append(cid_data)

## Format
model_comparisons = pd.DataFrame(model_comparisons)

## Check Sizes
model_comparison_support = pd.pivot_table(model_comparisons,
                                          index="source",
                                          columns="target",
                                          values=ALL_METRICS[0],
                                          aggfunc=lambda i: len(i.iloc[0][0])).fillna(0).astype(int)

## Output Support
LOGGER.info("Support Per Comparison:\n"+"~"*100)
LOGGER.info(model_comparison_support)

## Execute Comparisons
model_comparison_pivot = pd.pivot_table(model_comparisons,
                                        index="source",
                                        columns="target",
                                        values=ALL_METRICS,
                                        aggfunc=summarize_compare).fillna("")

## Output
for metric in ALL_METRICS:
    out_df = model_comparison_pivot[metric]
    out_ind = [i for i in DATASET_ORDER if i in out_df.index]
    out_col = [i for i in DATASET_ORDER if i in out_df.columns]
    out_df = out_df.loc[out_ind, out_col]
    LOGGER.info("\n" + "~"*100 + f"\n{metric}\n" + "~"*100)
    LOGGER.info(out_df)
