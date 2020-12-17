
########################
### Configuration
########################

## Choose Directories to Analyze
COMPARE_DIRS = {
    "LDA":"./data/results/depression/sample_size/clpsych_wolohan/LDA/",
    "PLDA":"./data/results/depression/sample_size/clpsych_wolohan/PLDA/",
}

## Metrics
METRIC_OPT = "f1"
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
        ## Optimize Model Parameters (Based on Source Data)
        sf_opt_C = sf_data.loc[(sf_data["domain"]=="source")&(sf_data["group"]=="development")].set_index("C")[METRIC_OPT].idxmax()
        sf_data = sf_data.loc[sf_data["C"]==sf_opt_C].reset_index(drop=True).copy()
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

########################
### Plot Results
########################

######## Sample Size

group = "development"
domain = "target"
metric = "f1"

sample_size = pd.pivot_table(scores_df.loc[(scores_df["domain"]==domain)&(scores_df["group"]==group)],
                             index=["target_sample_size"],
                             columns=["experiment","source_sample_size"],
                             values=metric,
                             aggfunc=np.mean)

lda_unstack = sample_size["LDA"].iloc[::-1]
plda_unstack = sample_size["PLDA"].iloc[::-1]

fig, ax = plt.subplots(1,2,figsize=(15,5), sharey=True)
for i, (u, cmap) in enumerate(zip([lda_unstack, plda_unstack],[plt.cm.Blues,plt.cm.Reds])):
    m = ax[i].imshow(u, cmap=cmap, aspect="auto", alpha=0.5, vmin=sample_size.min().min(), vmax=sample_size.max().max())
    ax[i].set_xticks(range(u.shape[1]))
    ax[i].set_xticklabels(["{:.0f}%".format(int(float("{:.1f}".format(i))*100)) for i in u.columns / u.columns.max()])
    ax[i].set_yticks(range(u.shape[0]))
    ax[i].set_yticklabels(["{:.0f}%".format(int(float("{:.1f}".format(i))*100)) for i in u.index / u.index.max()])
    for k, row in enumerate(u.values):
        for j, col in enumerate(row):
            if pd.isnull(col):
                continue
            ax[i].text(j, k, "{:.3f}".format(col)[1:], ha="center", va="center", color="black", fontsize=12, fontweight="bold" if col == np.max(u.values) else "normal")
    ax[i].set_xlabel("Source Sample Size", fontsize=18, fontweight="bold")
    if i == 0:
        ax[i].set_ylabel("Target Sample Size", fontsize=18, fontweight="bold")
ax[0].set_title("LDA", loc="left", fontweight="bold", fontstyle="italic", fontsize=20)
ax[1].set_title("PLDA", loc="left", fontweight="bold", fontstyle="italic", fontsize=20)
for a in ax:
    a.tick_params(labelsize=14)
fig.tight_layout()
plt.show()

######## Prior

# group = "development"
# domain = "target"
# metric = "f1"

# prior = pd.pivot_table(scores_df.loc[(scores_df["domain"]==domain)&(scores_df["group"]==group)],
#                index=["alpha","beta"],
#                columns=["experiment"],
#                values=metric,
#                aggfunc=np.mean)

# lda_unstack = prior.unstack()["LDA"].iloc[::-1]
# plda_unstack = prior.unstack()["PLDA"].iloc[::-1]

# fig, ax = plt.subplots(1,2,figsize=(15,5),sharey=True)
# for i, (u, cmap) in enumerate(zip([lda_unstack, plda_unstack],[plt.cm.Blues,plt.cm.Reds])):
#     m = ax[i].imshow(u, cmap=cmap, aspect="auto", alpha=0.5, vmin=prior.min().min(), vmax=prior.max().max())
#     ax[i].set_xticks( range(u.shape[1]))
#     ax[i].set_xticklabels(u.columns)
#     ax[i].set_yticks(range(u.shape[0]))
#     ax[i].set_yticklabels(u.index)
#     for k, row in enumerate(u.values):
#         for j, col in enumerate(row):
#             if pd.isnull(col):
#                 continue
#             ax[i].text(j, k, "{:.3f}".format(col)[1:], ha="center", va="center", color="black", fontsize=12, fontweight="bold" if col == np.max(u.values) else "normal")
#     ax[i].set_xlabel("$\\beta$", fontsize=18, fontweight="bold")
#     if i == 0:
#         ax[i].set_ylabel("$\\alpha$", fontsize=18, fontweight="bold", rotation=0, labelpad=20)
# ax[0].set_title("LDA", loc="left", fontweight="bold", fontstyle="italic", fontsize=20)
# ax[1].set_title("PLDA", loc="left", fontweight="bold", fontstyle="italic", fontsize=20)
# for a in ax:
#     a.tick_params(labelsize=14)
# fig.tight_layout()
# plt.show()

######## Latent Topics

# group = "development"
# domain = "target"
# metric = "f1"

# latent = pd.pivot_table(scores_df.loc[(scores_df["domain"]==domain)&(scores_df["group"]==group)],
#                index=["k_latent","k_per_label"],
#                columns=["experiment"],
#                values=metric,
#                aggfunc=np.mean)

# latent_lda = latent["LDA"].dropna()
# latent_plda_unstack = latent["PLDA"].unstack().T.iloc[::-1]

# fig, ax = plt.subplots(1,2,figsize=(15,5))
# ax[0].plot(latent_lda.index.levels[0], latent_lda.values, marker="o", linestyle="--", alpha=0.8, linewidth=3, markersize=10)
# ax[0].axhline(latent_plda_unstack.max().max(), linestyle="--", alpha=0.8, color="darkred", label="PLDA Max Score", linewidth=3)
# m = ax[1].imshow(latent_plda_unstack, cmap=plt.cm.Reds, aspect="auto", alpha=0.5)
# cbar = fig.colorbar(m, ax=ax[1])
# cbar.ax.tick_params(labelsize=14)
# ax[1].set_xticks( range(latent_plda_unstack.shape[1]))
# ax[1].set_xticklabels(latent_plda_unstack.columns)
# ax[1].set_yticks(range(latent_plda_unstack.shape[0]))
# ax[1].set_yticklabels(latent_plda_unstack.index)
# for i, row in enumerate(latent_plda_unstack.values):
#     for j, col in enumerate(row):
#         ax[1].text(j, i, "{:.3f}".format(col)[1:], ha="center", va="center", color="black", fontsize=12)
# ax[0].set_xlabel("# Latent Topics", fontsize=18, fontweight="bold")
# ax[0].set_ylabel(metric.replace("_"," ").title() if len(metric) > 3 else metric.upper(), fontsize=18, fontweight="bold")
# ax[1].set_xlabel("# Latent Topics", fontsize=18, fontweight="bold")
# ax[1].set_ylabel("# Topics\nPer Domain", fontsize=18, fontweight="bold")
# ax[0].spines["top"].set_visible(False)
# ax[0].spines["right"].set_visible(False)
# ax[0].legend(loc="lower right", fontsize=14)
# ax[0].set_title("LDA", loc="left", fontweight="bold", fontstyle="italic", fontsize=20)
# ax[1].set_title("PLDA", loc="left", fontweight="bold", fontstyle="italic", fontsize=20)
# for a in ax:
#     a.tick_params(labelsize=14)
# fig.tight_layout()
# plt.show()