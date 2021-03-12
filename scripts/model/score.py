
"""
Summarize Performance
"""

########################
### Configuration
########################

## Specify Plot Directory
PLOT_DIR = "./plots/classification/factor-sweep/lda/"

## Option 1: Choose Directories to Analyze
# COMPARE_DIRS = {
#     "clpsych_multitask":"./data/results/depression/hyperparameter-sweep/PLDA/*source-clpsych_deduped_target-multitask*/",
#     "multitask_clpsych":"./data/results/depression/hyperparameter-sweep/PLDA/*source-multitask_target-clpsych_deduped*/",
# }

## Option 2: Automatically Find Directories for Multiple Dataset Combinations
DATASETS = ["clpsych_deduped","multitask","wolohan","smhd"]
BASE_DIR = "./data/results/depression/factor-sweep/LDA/"
COMPARE_DIRS = {}
for source_ds in DATASETS:
    for target_ds in DATASETS:
        if source_ds == target_ds:
            continue
        if set([source_ds,target_ds]) == set(["rsdd","smhd"]):
            continue
        COMPARE_DIRS[f"{source_ds}-{target_ds}"] = f"{BASE_DIR}*source-{source_ds}_target-{target_ds}*/"

## Choose Metrics (Plotting + Optimization)
METRICS = ["f1","auc"]
OPT_METRICS = ["auc"]

## Choose Hyperparameters to Look at Together
JOINT_PARAMS = {
    # "prior":["alpha","beta"],
    # "latent_factors":["k_latent","k_per_label"]
}

## Cross Validation Flag
CROSS_VALIDATION = True

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
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

## Local
from mhlib.util.helpers import flatten
from mhlib.util.logging import initialize_logger

########################
### Globals
########################

## Plot Directory
if not os.path.exists(PLOT_DIR):
    _ = os.makedirs(PLOT_DIR)

## Logging
LOGGER = initialize_logger()

## All Metrics
GEN_COLS = ["domain","group","source","target","experiment","output_dir","fold","model_n"]
ALL_METRICS = ["auc","avg_precision","f1","precision","recall"]

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

def load_scores(directory,
                experiment_id,
                config):
    """

    """
    ## Check File
    if not os.path.exists(f"{directory}/classification/scores.csv"):
        return None
    ## Load Data
    scores = pd.read_csv(f"{directory}/classification/scores.csv").fillna("None")
    ## Add Metadata
    scores["experiment"] = experiment_id
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
    ## Add Joint Parameters
    for jp, jpl in JOINT_PARAMS.items():
        if not all(j in scores.columns for j in jpl):
            continue
        scores[jp] = scores[jpl].apply(tuple,axis=1).map(str)
    return scores

def bootstrap_ci(x,
                 n=100,
                 alpha=0.05,
                 aggfunc=np.mean):
    """

    """
    cache = np.zeros(n)
    theta_0 = aggfunc(x)
    for i in range(n):
        cache[i] = aggfunc(np.random.choice(x, len(x), replace=True))
    ## Compute Bounds
    q_l = np.nanpercentile(theta_0 - cache, q=100*alpha/2)
    q_u = np.nanpercentile(theta_0 - cache, q=100 - 100*alpha/2)
    ## Return
    q = np.array([theta_0 + q_l, theta_0, theta_0 + q_u])
    return tuple(q)

def plot_marginal_influence(scores_df,
                            vc,
                            vary_cols,
                            metric,
                            aggfunc=np.mean):
    """

    """
    ## Get Relevant Data Aggregations
    if vc not in JOINT_PARAMS.keys():
        group_cols = [v for v in vary_cols if v != vc and v not in JOINT_PARAMS.keys()]
    else:
        group_cols = [v for v in vary_cols if v != vc and v not in flatten(JOINT_PARAMS.values())]
    grouped_scores = scores_df.groupby(["domain","group"] + group_cols + [vc])[metric].agg([aggfunc, np.std])
    grouped_scores_avg = scores_df.groupby(["domain","group",vc])[metric].agg(bootstrap_ci).to_frame()
    for i in range(3):
        grouped_scores_avg[i] = grouped_scores_avg[metric].map(lambda j: j[i])
    ## Generate Plot
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    for d, domain in enumerate(["source","target"]):
        for g, group in enumerate(["train","development"]):
            pax = ax[d, g]
            pax_data = grouped_scores.loc[domain, group].reset_index().sort_values(vc)
            for opt, ind in  pax_data.groupby(group_cols).groups.items():
                opt_data = pax_data.loc[ind]
                offset = np.random.normal(0,0.01)
                pax.errorbar(np.arange(opt_data.shape[0])+offset,
                             opt_data[aggfunc.__name__].values,
                             yerr=opt_data["std"].values,
                             color="C0",
                             alpha=0.05,
                             zorder=-1)
            pax.errorbar(np.arange(opt_data.shape[0]),
                         grouped_scores_avg.loc[domain, group][1].values,
                         yerr=np.vstack([(grouped_scores_avg.loc[domain, group][1]-grouped_scores_avg.loc[domain, group][0]).values,
                                         (grouped_scores_avg.loc[domain, group][2]-grouped_scores_avg.loc[domain, group][1]).values]),
                         color="black",
                         linewidth=2,
                         zorder=1,
                         capsize=2)
            pax.set_title(f"{domain.title()} - {group.title()}")
            pax.spines["right"].set_visible(False)
            pax.spines["top"].set_visible(False)
            if pax.get_ylim()[0] < 0:
                pax.set_ylim(bottom=0)
            if g == 0:
                pax.set_ylabel(metric)
            if d == 1:
                pax.set_xlabel(f"{vc} Type")
            pax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return fig, ax

########################
### Load Results
########################

## Initialize Cache
scores_df = []

## Load Scores Iteratively
for exp_id, exp_dir in tqdm(COMPARE_DIRS.items(), total=len(COMPARE_DIRS), desc="Compare Categories", position=0):
    ## Identify Directories
    exp_id_dirs = glob(f"{exp_dir}")
    ## Iterate Through Directories
    for ed in tqdm(exp_id_dirs, desc="Comparison Directories", position=1, leave=False):
        ## Load Config
        with open(f"{ed}config.json","r") as the_file:
            ed_config = json.load(the_file)
        ## Cross Validation Scores
        if CROSS_VALIDATION:
            ## Load All Scores
            ed_fold_dirs = glob(f"{ed}fold-*/")
            for ed_fold_dir in ed_fold_dirs:
                sf_data = load_scores(ed_fold_dir, exp_id, ed_config)
                if sf_data is None:
                    LOGGER.warning(f"No data for {ed_fold_dir}")
                    continue
                ## Cache Scores
                scores_df.append(sf_data)
        else:
            ## Load Score Data
            sf_data = load_scores(ed, exp_id, ed_config)
            if sf_data is None:
                LOGGER.warning(f"No data for {ed_fold_dir}")
                continue
            ## Add General Fold
            sf_data["fold"] = 1
            ## Cache Scores
            scores_df.append(sf_data)

## Concatenate Scores
scores_df = pd.concat(scores_df).reset_index(drop=True)

########################
### Hyperparameter Analysis
########################

## Get Unique Experiments
experiments = scores_df["experiment"].unique()

## Iterate over Experiments
for e, experiment in enumerate(experiments):
    ## Logging
    LOGGER.info("\n" + "~"*50 + "\nStarting Hyperparameter Analysis for Experiment {}/{}: {}\n".format(e+1, len(experiments), experiment) + "~"*50)
    ## Isolate Experiment Data
    experiment_scores_df = scores_df.loc[scores_df["experiment"]==experiment].copy()
    ## Identify Parameters that Vary
    vary_cols = [v for v in experiment_scores_df.drop(ALL_METRICS + GEN_COLS, axis=1).columns.tolist() if len(experiment_scores_df[v].unique()) > 1]
    ## Display Unique Values
    LOGGER.info("\nParameter Groups:\n" + "~"*50)
    with open(f"{PLOT_DIR}{experiment}.params.txt","w") as the_file:
        for ctype, cvals in  experiment_scores_df[vary_cols].apply(set, axis=0).map(sorted).items():
            LOGGER.info("{}:{}".format(ctype,cvals))
            the_file.write(f"{ctype}: {cvals}\n")
    ## Visualize Marginal Influence of Each Parameter
    LOGGER.info("\nPlotting Marginal Parameter Influence\n" + "~"*50)
    for vc in tqdm(vary_cols, desc="Parameter Group", position=0):
        for metric in tqdm(METRICS, desc="Metric", position=1, leave=False):
            if metric not in experiment_scores_df.columns:
                continue
            fig, ax = plot_marginal_influence(experiment_scores_df,
                                              vc,
                                              vary_cols,
                                              metric,
                                              aggfunc=np.mean)
            fig.savefig(f"{PLOT_DIR}{experiment}_{vc}_{metric}.png", dpi=150)
            plt.close(fig)
    ## Optimal Model
    experiment_agg_scores = experiment_scores_df.groupby(["group","domain"] + vary_cols)[OPT_METRICS].agg([np.mean,np.std]).loc["development"]
    experiment_agg_scores["weighted_rank"] = experiment_agg_scores[[[o, "mean"] for o in OPT_METRICS]].sum(axis=1).rank(ascending=False)
    LOGGER.info("Displaying Top Models")
    for domain in ["source","target"]:
        top_exp_scores = experiment_agg_scores.loc[domain].sort_values("weighted_rank", ascending=True).head(10)
        LOGGER.info("~"*50 + f"\nDomain: {domain}\n"+"~"*50 + "\n"+top_exp_scores.to_string())