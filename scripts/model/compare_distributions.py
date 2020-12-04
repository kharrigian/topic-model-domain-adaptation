
#################
### Configuration
#################

## Output Directory
output_dir = "./plots/distributions/clpsych_wolohan/"

## Choose Source and Target Datasets
source = "clpsych"
target =  "wolohan"

## Analysis Parameters
embeddings_dim = 200
embeddings_size = 50000
embeddings_norm = "mean"
umap_params = {
    "metric":"cosine",
    "n_neighbors":30,
    "random_state":42
}

#################
### Imports
#################

## Standard Library
import os
import sys

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from umap import UMAP
from scipy import sparse, stats
import matplotlib.pyplot as plt
from mhlib.util.helpers import flatten
from mhlib.util.logging import initialize_logger

#################
### Globals
#################

## Logging
LOGGER = initialize_logger()

## Data Directories
DEPRESSION_DATA_DIR = f"./data/raw/depression/"

## Plot Directory
if not os.path.exists(output_dir):
    _ = os.makedirs(output_dir)

#################
### Helpers
#################

def load_data(data_dir):
    """

    """
    X = sparse.load_npz(f"{data_dir}data.npz")
    y = np.loadtxt(f"{data_dir}targets.txt")
    splits = [i.strip() for i in open(f"{data_dir}splits.txt","r")]
    filenames = [i.strip() for i in open(f"{data_dir}filenames.txt","r")]
    users = [i.strip() for i in open(f"{data_dir}users.txt","r")]
    terms = [i.strip() for i in open(f"{data_dir}vocab.txt","r")]
    return X, y, splits, filenames, users, terms

def load_glove(dim,
               resource_data_dir="./data/resources/",
               gV=100000):
    """

    """
    ## Check File
    glove_file = f"{resource_data_dir}glove.twitter.27B.{dim}d.txt"
    if not os.path.exists(glove_file):
        raise FileNotFoundError(f"Could not find glove embeddings: {glove_file}")
    ## Load Embeddings
    tokens = [None for _ in range(gV)]
    embeddings = np.zeros((gV,dim)) * np.nan
    with open(glove_file,"r") as the_file:
        for l, line in tqdm(enumerate(the_file),total=gV,desc="Loading Embeddings",file=sys.stdout):
            if l >= gV:
                break
            token, embedding = line.strip().split(" ",1)
            embedding = np.array(embedding.split()).astype(float)
            if embedding.shape[0] == dim:
                tokens[l] = token
                embeddings[l] = embedding
    ## Remove Flawed Embeddings
    mask = [i for i, j in enumerate(tokens) if j]
    tokens = [tokens[m] for m in mask]
    embeddings = embeddings[mask]
    return tokens, embeddings

def transform_X(X,
                X_vocab,
                embeddings,
                embeddings_vocab,
                norm=None):
    """

    """
    ## Construct Embeddings Transformer
    embeddings_word2idx = dict(zip(embeddings_vocab, range(embeddings.shape[0]))) 
    embeddings_alignment_mask = list(map(lambda xv: embeddings_word2idx.get(xv), X_vocab))
    embeddings_transformer = np.vstack(list(map(lambda ea: embeddings[ea] if ea is not None else np.zeros(embeddings.shape[1]), embeddings_alignment_mask)))
    ## Apply Transformation
    X_T = np.matmul(X.toarray(), embeddings_transformer)
    ## Find Match Mask
    Xv_match = (np.array(embeddings_alignment_mask) != None).astype(int).reshape(-1,1)
    n_match = np.matmul(X.toarray(), Xv_match)
    ## Apply Normalization
    if norm:
        if norm in ["l1","l2","max"]:
            X_T = normalize(X_T, norm=norm, axis=1)
        elif norm == "mean":
            X_T = np.divide(X_T,
                            n_match,
                            where=n_match>0,
                            out=np.zeros_like(X_T))
        else:
            raise ValueError(f"Norm `{norm}` not recognized")
    return X_T, n_match

def compute_odds(X, y, terms, alpha=1e-5):
    """

    """
    vc_df = pd.DataFrame(index=terms,
                                data=np.vstack([np.array(X[y==1].sum(axis=0))[0]+alpha,
                                                np.array(X[y==0].sum(axis=0))[0]+alpha,
                                                np.array((X[y==1]!=0).sum(axis=0))[0]+alpha,
                                                np.array((X[y==0]!=0).sum(axis=0))[0]+alpha]).T,
                                columns=["freq_depression","freq_control","users_depression","users_control"])
    vc_df["freq_depression"] /= vc_df["freq_depression"].sum()
    vc_df["freq_control"] /= vc_df["freq_control"].sum()
    vc_df["users_depression"] /= y.sum()
    vc_df["users_control"] /= (y!=1).sum()
    vc_df["freq_odds"] = np.log(vc_df["freq_depression"] / vc_df["freq_control"])
    vc_df["users_odds"] = np.log(vc_df["users_depression"] / vc_df["users_control"])
    return vc_df

#################
### Load Data
#################

LOGGER.info("Loading Data")

## Load Data
X_source, y_source, splits_source, filenames_source, users_source, terms_source = load_data(f"{DEPRESSION_DATA_DIR}{source}/")
X_target, y_target, splits_target, filenames_target, users_target, terms_target = load_data(f"{DEPRESSION_DATA_DIR}{target}/")

#################
### Visualize Vocabulary Differences
#################

LOGGER.info("Examining Vocabulary Differences")

## Compute Vocabulary Frequency
v_source_df = pd.DataFrame(index=terms_source,
                           data=np.vstack([np.array(X_source.sum(axis=0))[0], np.array((X_source!=0).sum(axis=0))[0]]).T,
                           columns=["n_freq","n_users"])
v_target_df = pd.DataFrame(index=terms_target,
                           data=np.vstack([np.array(X_target.sum(axis=0))[0], np.array((X_target!=0).sum(axis=0))[0]]).T,
                           columns=["n_freq","n_users"])
v_df = pd.merge(v_source_df, v_target_df, suffixes=("_source","_target"), how="outer", left_index=True, right_index=True)

## Compute Overlap
v_overlap = np.array([[v_source_df.shape[0], v_df.dropna().shape[0]],
                      [v_df.dropna().shape[0], v_target_df.shape[0]]])
v_overlap_normed = v_overlap / v_overlap.diagonal().reshape(-1,1)

## Plot Frequencies
fig, ax = plt.subplots(1,3,figsize=(10,5.8))
ax[0].scatter(v_df["n_freq_source"].fillna(0),
              v_df["n_freq_target"].fillna(0),
              alpha=0.1,
              label="Spearman R: {:.3f}".format(stats.spearmanr(v_df["n_freq_source"].fillna(0), v_df["n_freq_target"].fillna(0))[0]),
              s=10)
ax[1].scatter(v_df["n_users_source"].fillna(0),
              v_df["n_users_target"].fillna(0),
              label="Spearman R: {:.3f}".format(stats.spearmanr(v_df["n_users_source"].fillna(0), v_df["n_users_target"].fillna(0))[0]),
              alpha=0.1,
              s=10)
ax[2].imshow(v_overlap_normed, cmap=plt.cm.Blues, aspect="auto", interpolation="nearest", alpha=0.7)
for i, row in enumerate(v_overlap):
    for j, val in enumerate(row):
        ax[2].text(j, i, "{:,d}".format(val), ha="center", va="center")
for i, l in enumerate(["Frequency","Users"]):
    ax[i].set_xscale("symlog")
    ax[i].set_yscale("symlog")
    ax[i].set_xlabel(f"Source {l}", fontweight="bold")
    ax[i].set_ylabel(f"Target {l}", fontweight="bold")
    ax[i].legend(loc="upper left", frameon=True)
for i in range(3):
    ax[i].spines["top"].set_visible(False)
    ax[i].spines["right"].set_visible(False)
ax[2].set_xticks([0,1]); ax[2].set_xticklabels(["Source","Target"], fontweight="bold")
ax[2].set_yticks([0,1]); ax[2].set_yticklabels(["Source","Target"], fontweight="bold")
fig.tight_layout()
fig.savefig(f"{output_dir}vocab_frequency.png",dpi=300)
plt.close(fig)

## Compute Log Odds
vc_source_df = compute_odds(X_source, y_source, terms_source)
vc_target_df = compute_odds(X_target, y_target, terms_target)
vc_df = pd.merge(vc_source_df, vc_target_df, left_index=True, right_index=True, how="outer", suffixes=("_source","_target"))

## Plot Log Odds Comparsison
fig, ax = plt.subplots(1,2,figsize=(10,5.8))
ax[0].scatter(vc_df.dropna()["freq_odds_source"],
              vc_df.dropna()["freq_odds_target"],
              alpha=0.1,
              s=10)
ax[1].scatter(vc_df.dropna()["users_odds_source"],
              vc_df.dropna()["users_odds_target"],
              alpha=0.1,
              s=10)
for i, l in enumerate(["Frequency","Users"]):
    ax[i].set_xlabel(f"Source {l} Odds", fontweight="bold")
    ax[i].set_ylabel(f"Target {l} Odds", fontweight="bold")
    ax[i].spines["top"].set_visible(False)
    ax[i].spines["right"].set_visible(False)
    ax[i].axvline(0,color="black",alpha=0.1,linestyle="--",zorder=-1)
    ax[i].axhline(0,color="black",alpha=0.1,linestyle="--",zorder=-1)
fig.tight_layout()
fig.savefig(f"{output_dir}log_odds.png",dpi=300)
plt.close(fig)

#################
### Visualize Semantic Differences
#################

LOGGER.info("Comparing Semantic Distributions")

## Load Embeddings
embeddings_vocab, embeddings = load_glove(embeddings_dim, gV=embeddings_size)

## Apply Transformations to get GloVe Representations
LOGGER.info("Transforming Document-Term Matrices into Embedding Space")
X_T_source, n_match_source = transform_X(X_source, terms_source, embeddings, embeddings_vocab, embeddings_norm)
X_T_target, n_match_target = transform_X(X_target, terms_target, embeddings, embeddings_vocab, embeddings_norm)

## Concatenate Data
X_all = np.vstack([X_T_source, X_T_target])
nn_mask = ~(X_all == 0).all(axis=1)

## Fit Model
LOGGER.info("Projecting Data using UMAP")
projector = UMAP(verbose=True,
                 **umap_params)
X_all_proj = projector.fit_transform(X_all)

## Show Projection
fig, ax = plt.subplots(figsize=(10,5.6))
ax.scatter([],[],color="navy",label="Source",s=30,alpha=.5)
ax.scatter([],[],color="darkred",label="Target",s=30,alpha=.5)
ax.scatter(X_all_proj[nn_mask][:,0],
           X_all_proj[nn_mask][:,1],
           alpha=0.1,
           s=10,
           c=["navy"] * nn_mask[:y_source.shape[0]].sum() + ["darkred"]*nn_mask[y_source.shape[0]:].sum())
ax.legend(loc="lower right", frameon=False, fontsize=20)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{output_dir}umap.png",dpi=300)
plt.close(fig)

LOGGER.info("Script complete!")