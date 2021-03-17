
"""
Compare sensitivity of hyperparameters between LDA and PLDA
"""

########################
### Configuration
########################

## Specify Plot Directory
PLOT_DIR = "./plots/classification/factor-sweep/comparsion/"

## Option 2: Automatically Find Directories for Multiple Dataset Combinations
DATASETS = ["clpsych_deduped","multitask","wolohan","smhd"]
BASE_DIR = "./data/results/depression/factor-sweep/"
COMPARE_DIRS = {}
for model in ["LDA","PLDA"]:
    COMPARE_DIRS[model] = {}
    for source_ds in DATASETS:
        for target_ds in DATASETS:
            if source_ds == target_ds:
                continue
            if set([source_ds,target_ds]) == set(["rsdd","smhd"]):
                continue
            COMPARE_DIRS[model][f"{source_ds}-{target_ds}"] = f"{BASE_DIR}{model}/*source-{source_ds}_target-{target_ds}*/"

## Choose Metrics (Plotting + Optimization)
METRICS = ["auc"]

## Choose Hyperparameters to Look at Together
JOINT_PARAMS = {
    # "prior":["alpha","beta"],
    # "latent_factors":["k_latent","k_per_label"]
}

## Cross Validation Flag
CROSS_VALIDATION = False

## Filters
FILTERS = {
    "norm":["l2"],
}

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
GEN_COLS = ["domain","group","source","target","experiment","output_dir","fold","model_n","model_type"]
ALL_METRICS = ["auc","avg_precision","f1","precision","recall"]

########################
### Helpers
########################

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
        scores[jp] = scores[jpl].apply(tuple,axis=1)
    ## Filtering
    for fkey, fvals in FILTERS.items():
        fval_set = set(fvals)
        scores = scores.loc[scores[fkey].isin(fval_set)]
    if len(scores) == 0:
        return None
    else:
        scores = scores.reset_index(drop=True).copy()
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
                            vary_cols_all,
                            metric,
                            aggfunc=np.mean,
                            ax=None,
                            color="C0",
                            label=None):
    """

    """
    ## Get Relevant Data Aggregations
    if vc not in JOINT_PARAMS.keys():
        group_cols = [v for v in vary_cols_all if v != vc and v not in JOINT_PARAMS.keys()]
    else:
        group_cols = [v for v in vary_cols_all if v != vc and v not in flatten(JOINT_PARAMS.values())]
    grouped_scores = scores_df.groupby(["domain","group"] + group_cols + [vc])[metric].agg([aggfunc, np.std])
    grouped_scores_all = scores_df.groupby(["domain","group"] + vary_cols_all)[metric].mean()
    grouped_scores_avg = scores_df.groupby(["domain","group",vc])[metric].agg(bootstrap_ci).to_frame()
    for i in range(3):
        grouped_scores_avg[i] = grouped_scores_avg[metric].map(lambda j: j[i])
    ## Generate Plot
    if ax is None:
        fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    else:
        fig = None
    for d, domain in enumerate(["source","target"]):
        for g, group in enumerate(["train","development"]):
            pax = ax[d, g]
            pax_data = grouped_scores.loc[domain, group].reset_index().sort_values(vc)
            for opt, ind in pax_data.groupby(group_cols).groups.items():
                opt_data = pax_data.loc[ind]
                offset = np.random.normal(0,0.01)
                pax.errorbar(np.arange(opt_data.shape[0])+offset,
                             opt_data[aggfunc.__name__].values,
                             yerr=opt_data["std"].values if not np.isnan(opt_data["std"].values).all() else None,
                             color=color,
                             alpha=0.05,
                             zorder=-1)
            pax.errorbar(np.arange(opt_data.shape[0]),
                         grouped_scores_avg.loc[domain, group][1].values,
                         yerr=np.vstack([(grouped_scores_avg.loc[domain, group][1]-grouped_scores_avg.loc[domain, group][0]).values,
                                         (grouped_scores_avg.loc[domain, group][2]-grouped_scores_avg.loc[domain, group][1]).values]),
                         color=color,
                         linewidth=2,
                         zorder=1,
                         capsize=2,
                         label=label)
            pax_max = grouped_scores_all.loc[domain, group].max()
            pax.axhline(pax_max,
                        color=color,
                        linestyle="--",
                        linewidth=2,
                        alpha=0.9)
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
    return fig, ax

########################
### Load Results
########################

## Initialize Cache
scores_df = []

## Load Scores Iteratively
for model_type, model_dirs in COMPARE_DIRS.items():
    LOGGER.info(f"Loading {model_type} Scores")
    for exp_id, exp_dir in tqdm(model_dirs.items(), total=len(model_dirs), desc="Compare Categories", position=0):
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
                    sf_data["model_type"] = model_type
                    scores_df.append(sf_data)
            else:
                ## Load Score Data
                sf_data = load_scores(ed, exp_id, ed_config)
                if sf_data is None:
                    LOGGER.warning(f"No data for {ed}")
                    continue
                ## Add General Fold
                sf_data["fold"] = 1
                sf_data["model_type"] = model_type
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
    exp_lda_scores_df = scores_df.loc[(scores_df["experiment"]==experiment)&
                                      (scores_df["model_type"]=="LDA")].copy()
    exp_plda_scores_df = scores_df.loc[(scores_df["experiment"]==experiment)&
                                       (scores_df["model_type"]=="PLDA")].copy()
    ## Identify Parameters that Vary
    vary_cols_lda = [v for v in exp_lda_scores_df.drop(ALL_METRICS + GEN_COLS, axis=1).columns.tolist() if len(exp_lda_scores_df[v].unique()) > 1]
    vary_cols_plda = [v for v in exp_plda_scores_df.drop(ALL_METRICS + GEN_COLS, axis=1).columns.tolist() if len(exp_plda_scores_df[v].unique()) > 1]
    vary_cols = sorted(set(vary_cols_lda) & set(vary_cols_plda))
    vary_cols_all = sorted(set(vary_cols_lda) | set(vary_cols_plda))
    ## Display Unique Values
    LOGGER.info("\nParameter Groups:\n" + "~"*50)
    for ctype, cvals in exp_lda_scores_df.append(exp_plda_scores_df)[vary_cols].apply(set, axis=0).map(sorted).items():
        LOGGER.info("{}: {}".format(ctype,cvals))
    ## Visualize Marginal Influence of Each Parameter
    LOGGER.info("\nPlotting Marginal Parameter Influence\n" + "~"*50)
    for vc in tqdm(vary_cols, desc="Parameter Group", position=0):
        for metric in tqdm(METRICS, desc="Metric", position=1, leave=False):
            if metric not in scores_df.columns:
                continue
            fig, ax = plt.subplots(2,2,figsize=(10,5.8))
            for d, df in enumerate([exp_lda_scores_df, exp_plda_scores_df]):
                _, ax = plot_marginal_influence(df,
                                                vc,
                                                vary_cols,
                                                vary_cols_all,
                                                metric,
                                                aggfunc=np.mean,
                                                ax=ax,
                                                color=f"C{d}",
                                                label=["LDA","PLDA"][d])
            for a in ax:
                for b in a:
                    b.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(f"{PLOT_DIR}{experiment}_{vc}_{metric}.png", dpi=150)
            plt.close(fig)
