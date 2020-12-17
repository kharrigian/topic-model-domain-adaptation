
########################
### Configuration
########################

## Choose Experiment Directory
EXPERIMENT_DIR = "./data/results/synthetic/DGP-CovariateShift/"

## Metrics to Plot
METRIC_VARS = ["f1","auc"]

## Plot Directory
PLOT_DIR = "./plots/classification/synthetic/DGP-CovariateShift/"

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

## Experiments
experiments = glob(f"{EXPERIMENT_DIR}*scores.json")

## Load and Parse Scores
scores_df = []
for score_file in experiments:
    with open(score_file,"r") as the_file:
        exp_scores = json.load(the_file)
    with open(score_file.replace(".scores.",".config."),"r") as the_file:
        exp_config = json.load(the_file)
    ## Score Difference
    exp_scores["Difference"] = {}
    exp_scores["Relative Difference"] = {}
    for eval_set in ["Source","Target","Overall"]:
        lda_perf = exp_scores["LDA"][eval_set]
        plda_perf = exp_scores["PLDA"][eval_set]
        perf_diff = dict((s, plda_perf[s]-lda_perf[s]) for s in ["f1","auc"])
        rel_perf_diff = {"f1":perf_diff["f1"] / max(lda_perf["f1"], 1e-5) * 100,
                         "auc":perf_diff["auc"] / max(lda_perf["auc"], 1e-5) * 100
        }
        exp_scores["Difference"][eval_set] = perf_diff
        exp_scores["Relative Difference"][eval_set] = rel_perf_diff
    ## Cache Params
    for model, model_res in exp_scores.items():
        for eval_set, eval_scores in model_res.items():
            eval_flat = eval_scores.copy()
            eval_flat.update({"model":model,"domain":eval_set})
            for ck, cv in exp_config.items():
                if ck in ["output_dir","run_id"]:
                    continue
                if ck == "theta":
                    eval_flat["theta_source_ratio"] = cv[0][0] / cv[0][2]
                    eval_flat["theta_target_ratio"] = cv[1][1] / cv[1][2]
                    eval_flat["theta_source"] = cv[0]
                    eval_flat["theta_target"] = cv[1]
                elif ck == "coef":
                    eval_flat["coef_source"] = cv[0]
                    eval_flat["coef_target"] = cv[1]
                else:
                    eval_flat[ck] = cv
            scores_df.append(eval_flat)
scores_df = pd.DataFrame(scores_df).reset_index(drop=True)

## Ratio Formatter
ratio_formatter = lambda x: str(int(x)) if x >= 1 else "{:.2f}".format(x)[1:]

## Initialize Plot Directory
if not os.path.exists(PLOT_DIR):
    _ = os.makedirs(PLOT_DIR)

## Generate Plots
for domain in ["Target","Source","Overall"]:
    for metric in METRIC_VARS:
        ## Separate Scores
        lda = scores_df.loc[(scores_df["model"]=="LDA")&(scores_df["domain"]==domain)]
        plda = scores_df.loc[(scores_df["model"]=="PLDA")&(scores_df["domain"]==domain)]
        diff = scores_df.loc[(scores_df["model"]=="Difference")&(scores_df["domain"]==domain)]

        ## Plot Scores
        fig, axes = plt.subplots(1, 3, figsize=(15,5), sharex=True, sharey=True)
        for d, (data, score, cmap) in enumerate(zip([lda,plda,diff],
                                                    ["LDA","PLDA","Difference"],
                                                    [plt.cm.Blues, plt.cm.Reds, plt.cm.coolwarm])):
            ax = axes[d]
            data_pivot = pd.pivot_table(data,
                                        index=["theta_target_ratio"],
                                        columns=["theta_source_ratio"],
                                        values=metric,
                                        aggfunc=np.mean).iloc[::-1]
            bound = np.max(np.abs(data_pivot).values)
            ax.imshow(data_pivot,
                    cmap=cmap,
                    vmin=-bound if d == 2 else np.min(data_pivot.values),
                    vmax=bound if d == 2 else np.max(data_pivot.values),
                    aspect="auto",
                    interpolation="nearest",
                    alpha=0.8
                    )
            for i, row in enumerate(data_pivot.values):
                for j, cell in enumerate(row):
                    prefix = "" if cell >0 else "-"
                    cell = prefix + "{:.3f}".format(abs(cell))[1:]
                    ax.text(j, i, cell, ha="center", va="center", fontsize=10)
            ax.set_title(score, loc="left", fontstyle="italic", fontsize=20, fontweight="bold")
            ax.set_xticks(range(data_pivot.shape[1]))
            ax.set_xticklabels([ratio_formatter(i) for i in data_pivot.columns])
            ax.set_yticks(range(data_pivot.shape[0]))
            ax.set_yticklabels([ratio_formatter(i) for i in data_pivot.index])
            ax.set_xlabel("$x_{S}$", fontsize=18)
            ax.tick_params(labelsize=12)
        axes[0].set_ylabel("$x_{T}$", fontsize=18, rotation=0)
        fig.tight_layout()
        fig.savefig(f"{PLOT_DIR}{domain}_{metric}.pdf")
        plt.close(fig)