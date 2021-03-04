
"""
Summarize Performance
"""

########################
### Configuration
########################

## Choose Directories to Analyze
COMPARE_DIRS = {
    "clpsych_multitask":"./data/results/depression/hyperparameter-sweep/PLDA/*source-clpsych_deduped_target-multitask*/",
    "multitask_clpsych":"./data/results/depression/hyperparameter-sweep/PLDA/*source-multitask_target-clpsych_deduped*/",
}

## Plot Directory
PLOT_DIR = "./plots/classification/test/"

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

########################
### Globals
########################

## Plot Directory
if not os.path.exists(PLOT_DIR):
    _ = os.makedirs(PLOT_DIR)

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
    return scores

def plot_marginal_influence(scores_df,
                            vc,
                            vary_cols,
                            metric,
                            aggfunc=np.mean):
    """

    """
    ## Get Relevant Data Aggregations
    group_cols = [v for v in vary_cols if v != vc]
    grouped_scores = scores_df.groupby(["domain","group"] + group_cols + [vc])[metric].agg([aggfunc, np.std])
    grouped_scores_avg = scores_df.groupby(["domain","group",vc])[metric].agg([aggfunc,np.std])
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
                             alpha=0.01,
                             zorder=-1)
            pax.errorbar(np.arange(opt_data.shape[0]),
                         grouped_scores_avg.loc[domain, group][aggfunc.__name__].values,
                         yerr=grouped_scores_avg.loc[domain, group]["std"].values,
                         color="black",
                         linewidth=3,
                         zorder=1,
                         capsize=3)
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
                    print(f"No data for {ed_fold_dir}")
                    continue
                ## Cache Scores
                scores_df.append(sf_data)
        else:
            ## Load Score Data
            sf_data = load_scores(ed, exp_id, ed_config)
            if sf_data is None:
                print(f"No data for {ed_fold_dir}")
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
    print("\n" + "~"*50 + "\nStarting Hyperparameter Analysis for Experiment {}/{}: {}\n".format(e+1, len(experiments), experiment) + "~"*50)
    ## Isolate Experiment Data
    experiment_scores_df = scores_df.loc[scores_df["experiment"]==experiment].copy()
    ## Identify Parameters that Vary
    all_metrics = ["auc","avg_precision","f1","precision","recall"]
    gen_cols = ["domain","group","source","target","experiment","output_dir","fold","model_n"]
    vary_cols = [v for v in experiment_scores_df.drop(all_metrics + gen_cols, axis=1).columns.tolist() if len(experiment_scores_df[v].unique()) > 1]
    ## Display Unique Values
    print("\nParameter Groups:\n" + "~"*50)
    for ctype, cvals in  experiment_scores_df[vary_cols].apply(set, axis=0).map(sorted).items():
        print(ctype,":",cvals)
    ## Visualize Marginal Influence of Each Parameter
    print("\nPlotting Marginal Parameter Influence\n" + "~"*50)
    for vc in tqdm(vary_cols, desc="Parameter Group", position=0):
        for metric in tqdm(all_metrics, desc="Metric", position=1, leave=False):
            if metric not in experiment_scores_df.columns:
                continue
            fig, ax = plot_marginal_influence(experiment_scores_df,
                                              vc,
                                              vary_cols,
                                              metric,
                                              aggfunc=np.mean)
            fig.savefig(f"{PLOT_DIR}{experiment}_{vc}_{metric}.png", dpi=150)
            plt.close(fig)

####################
### Optimal Models ##TODO
####################