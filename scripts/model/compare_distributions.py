
#################
### Configuration
#################

## Output Directory
output_dir = "./plots/distributions/clpsych_wolohan/"

## Choose Source and Target Datasets
source = "clpsych"
target = "wolohan"

## Analysis Parameters
min_term_freq = 5
min_user_freq = 5
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
import tomotopy as tp
from scipy import sparse, stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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

## Names
DATASET_NAMES = {
    "clpsych":"CLPsych",
    "wolohan":"Topic-Restricted",
    "multitask":"Multitask"
}

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
                                columns=["count_depression","count_control","count_users_depression","count_users_control"])
    vc_df["prob_depression"] = vc_df["count_users_depression"] / vc_df[["count_users_depression","count_users_control"]].sum(axis=1)
    vc_df["freq_depression"] = vc_df["count_depression"] / vc_df["count_depression"].sum()
    vc_df["freq_control"] = vc_df["count_control"] / vc_df["count_control"].sum()
    vc_df["freq_users_depression"] = vc_df["count_users_depression"] / y.sum()
    vc_df["freq_users_control"] = vc_df["count_users_control"] / (y!=1).sum()
    vc_df["term_odds"] = np.log(vc_df["freq_depression"] / vc_df["freq_control"])
    vc_df["users_odds"] = np.log(vc_df["freq_users_depression"] / vc_df["freq_users_control"])
    return vc_df

def align_data(X_source, X_target, vocab_source, vocab_target, how="outer"):
    """

    """
    ## Mappings
    source2ind = dict(zip(vocab_source, range(len(vocab_source))))
    target2ind = dict(zip(vocab_target, range(len(vocab_target))))
    ## Vocabulary Alignment
    vocab_source_unique = set(vocab_source)
    vocab_target_unique = set(vocab_target)
    if how == "outer":
        vocab = sorted(vocab_source_unique | vocab_target_unique)
    elif how == "inner":
        vocab = sorted(vocab_source_unique & vocab_target_unique)
    elif how == "source":
        vocab = vocab_source
    elif how == "target":
        vocab = vocab_target
    else:
        raise ValueError("`how` not recognized")
    overlap = sorted(set(vocab) & (vocab_source_unique & vocab_target_unique))
    source_only = sorted(set(vocab) & (vocab_source_unique - vocab_target_unique))
    target_only = sorted(set(vocab) & (vocab_target_unique - vocab_source_unique))
    vocab = overlap + source_only + target_only
    ## Update Matrices
    Xs = sparse.hstack([X_source[:,[source2ind[o] for o in overlap]],
                        X_source[:,[source2ind[o] for o in source_only]],
                        sparse.csr_matrix((X_source.shape[0],len(target_only)))])
    Xt = sparse.hstack([X_target[:,[target2ind[o] for o in overlap]],
                        sparse.csr_matrix((X_target.shape[0],len(source_only))),
                        X_target[:,[target2ind[o] for o in target_only]]])
        
    return Xs.tocsr(), Xt.tocsr(), vocab

def _rebalance(X,
               y,
               class_ratio=None,
               random_seed=42):
    """

    """
    ## Set Seed
    if random_seed is not None:
        np.random.seed(random_seed)
    ## Case 0: No Action
    if class_ratio is None:
        return X, y
    ## Get Data Distribution
    n0 = (y==0).sum()
    n1 = (y==1).sum()
    ## Target Control Size
    target_control_size = int(n1 * class_ratio[1] / class_ratio[0])
    ## Case 1: Keep Target Class Fixed
    if n0 >= target_control_size:
        control_sample = np.random.choice(np.where(y==0)[0], target_control_size, replace=False)
        target_sample = np.where(y==1)[0]
    ## Case 2: Downsample Everything So That Ratio Is Preserved
    else:
        n_target = n1
        while (n_target * class_ratio[1]) > n0:
            n_target -= 1
        n_control = class_ratio[1] * n_target
        control_sample = np.random.choice(np.where(y==0)[0], n_control, replace=False)
        target_sample = np.random.choice(np.where(y==1)[0], n_target, replace=False)
    ## Apply Mask
    mask = sorted(list(control_sample) + list(target_sample))
    X = X[mask].copy()
    y = y[mask].copy()
    return X, y

def _downsample(X,
                y,
                sample_size=None,
                random_seed=42):
    """

    """
    ## Set Seed
    if random_seed is not None:
        np.random.seed(random_seed)
    ## Case 0: No Action
    if sample_size is None:
        return X, y
    ## Create Sample
    mask = np.random.choice(X.shape[0], min(sample_size, X.shape[0]), replace=False)
    X = X[mask].copy()
    y = y[mask].copy()
    return X, y

def sample_data(X,
                y,
                class_ratio=None,
                sample_size=None,
                random_seed=42):
    """

    """
    ## Rebalance Data
    X, y = _rebalance(X, y, class_ratio, random_seed)
    ## Downsample Data
    X, y = _downsample(X, y, sample_size, random_seed)
    return X, y

## Helper Function for Converting Count Data
term_expansion = lambda x, vocab: flatten([[t]*int(i) for i, t in zip(x.toarray()[0], vocab)])

def generate_corpus(Xs, Xt, vocab, source=True, target=True, ys=None, yt=None):
    """

    """
    corpus = tp.utils.Corpus()
    missing = {"source":[],"target":[]}
    for i, x in tqdm(enumerate(Xs), total=Xs.shape[0], desc="Adding Source Documents", file=sys.stdout):
        if source:
            x_flat = term_expansion(x, vocab)
        else:
            x_flat = []
        if len(x_flat) == 0:
            missing["source"].append(i)
            continue
        labels = ["source"]
        if ys is not None:
            labels.append({0:"control",1:"depression"}.get(ys[i]))
        corpus.add_doc(term_expansion(x, vocab),labels=labels)
    for i, x in tqdm(enumerate(Xt), total=Xt.shape[0], desc="Adding Target Documents", file=sys.stdout):
        if target:
            x_flat = term_expansion(x, vocab)
        else:
            x_flat = []
        if len(x_flat) == 0:
            missing["target"].append(i)
            continue
        labels = ["target"]
        if yt is not None:
            labels.append({0:"control",1:"depression"}.get(yt[i]))
        corpus.add_doc(x_flat,labels=labels)
    return corpus, missing


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
v_df = pd.merge(v_source_df,
                v_target_df,
                suffixes=("_source","_target"),
                how="outer",
                left_index=True,
                right_index=True)

## Compute Overlap
v_overlap = np.array([[v_source_df.shape[0], v_df.dropna().shape[0]],
                      [v_df.dropna().shape[0], v_target_df.shape[0]]])
v_overlap_normed = v_overlap / v_overlap.diagonal().reshape(-1,1)

## Filtering
for domain in ["source","target"]:
    for prefix, criteria in zip(["n_freq","n_users"],[min_term_freq, min_user_freq]):
        v_df = v_df.loc[v_df[f"{prefix}_{domain}"].fillna(0) >= criteria].copy()

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
    ax[i].set_xlabel("{} {}".format(DATASET_NAMES[source], l), fontweight="bold")
    ax[i].set_ylabel("{} {}".format(DATASET_NAMES[target], l), fontweight="bold")
    ax[i].legend(loc="upper left", frameon=True)
for i in range(3):
    ax[i].spines["top"].set_visible(False)
    ax[i].spines["right"].set_visible(False)
ax[2].set_xticks([0,1]); ax[2].set_xticklabels([DATASET_NAMES[source],DATASET_NAMES[target]], fontweight="bold")
ax[2].set_yticks([0,1]); ax[2].set_yticklabels([DATASET_NAMES[source],DATASET_NAMES[target]], fontweight="bold")
fig.tight_layout()
fig.savefig(f"{output_dir}vocab_frequency.png",dpi=300)
fig.savefig(f"{output_dir}vocab_frequency.pdf")
plt.close(fig)

## Compute Log Odds
vc_source_df = compute_odds(X_source, y_source, terms_source)
vc_target_df = compute_odds(X_target, y_target, terms_target)
vc_df = pd.merge(vc_source_df,
                 vc_target_df,
                 left_index=True,
                 right_index=True,
                 how="outer",
                 suffixes=("_source","_target"))

## Filtering
for domain in ["source","target"]:
    for label in ["depression","control"]:
        for prefix, criteria in zip(["count","count_users"],[min_term_freq, min_user_freq]):
            vc_df = vc_df.loc[vc_df[f"{prefix}_{label}_{domain}"].fillna(0) >= criteria].copy()

## Plot Log Odds Comparsison
fig, ax = plt.subplots(1,2,figsize=(10,5.8))
ax[0].scatter(vc_df.dropna()["term_odds_source"],
              vc_df.dropna()["term_odds_target"],
              alpha=0.1,
              s=10,
              label="$\\log(\\frac{p(x|depression)}{p(x|control)})$")
ax[1].scatter(vc_df.dropna()["users_odds_source"],
              vc_df.dropna()["users_odds_target"],
              alpha=0.1,
              s=10,
              label="$\\log(\\frac{p(x|depression)}{p(x|control)})$")
for i, l in enumerate(["Frequency","Users"]):
    ax[i].set_xlabel("{} {} Odds".format(DATASET_NAMES[source], l), fontweight="bold")
    ax[i].set_ylabel("{} {} Odds".format(DATASET_NAMES[target], l), fontweight="bold")
    ax[i].legend(loc="upper right", frameon=True)
    ax[i].spines["top"].set_visible(False)
    ax[i].spines["right"].set_visible(False)
    ax[i].axvline(0,color="black",alpha=0.1,linestyle="--",zorder=-1)
    ax[i].axhline(0,color="black",alpha=0.1,linestyle="--",zorder=-1)
fig.tight_layout()
fig.savefig(f"{output_dir}log_odds.png",dpi=300)
fig.savefig(f"{output_dir}log_odds.pdf")
plt.close(fig)

## Compare P(y|word)
k_top = 40
fig, ax = plt.subplots(1,3,figsize=(10,5.8))
vc_df["prob_depression_source"].dropna().nlargest(k_top).iloc[::-1].plot.barh(ax=ax[0])
vc_df["prob_depression_target"].dropna().nlargest(k_top).iloc[::-1].plot.barh(ax=ax[1])
ax[2].scatter(vc_df["prob_depression_source"],
              vc_df["prob_depression_target"],
              alpha=0.1,
              s=10,
              label="Spearman R: {:.4f}".format(stats.spearmanr(vc_df["prob_depression_source"],vc_df["prob_depression_target"])[0]))
for a in ax:
    a.spines["right"].set_visible(False)
    a.spines["top"].set_visible(False)
for i in range(2):
    ax[i].set_xlabel("Pr(Depression | term)", fontweight="bold")
    ax[i].set_title([DATASET_NAMES[source],DATASET_NAMES[target]][i], fontweight="bold")
ax[2].set_title("Comparison", fontweight="bold")
ax[2].set_xlabel("{} Pr(Depression|term)".format(DATASET_NAMES[source]), fontweight="bold")
ax[2].set_ylabel("{} Pr(Depression|term)".format(DATASET_NAMES[target]), fontweight="bold")
ax[2].legend(loc="upper right")
fig.tight_layout()
fig.savefig(f"{output_dir}probability_depression.png",dpi=300)
fig.savefig(f"{output_dir}probability_depression.pdf")
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
ax.scatter([],[],color="navy",label=DATASET_NAMES[source],s=30,alpha=.5)
ax.scatter([],[],color="darkred",label=DATASET_NAMES[target],s=30,alpha=.5)
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
fig.savefig(f"{output_dir}umap.pdf")
plt.close(fig)

#################
### Topic Comparison
#################

## Downsample
X_source, y_source = sample_data(X_source, y_source, [1,1], 500)
X_target, y_target = sample_data(X_target, y_target, [1,1], 500)

## Align Vocabulary Spaces
X_source, X_target, vocab = align_data(X_source, X_target, terms_source, terms_target, "outer")

## Identifity Vocabulary Filter
vocab_mask = np.ones(len(vocab),dtype=int)
for x in [X_source,X_target]:
    vocab_mask[np.where(x.sum(axis=0) < min_term_freq)[1]] = 0
    vocab_mask[np.where((x!=0).sum(axis=0) < min_user_freq)[1]] = 0

## Apply Vocabulary Filtering
vocab_mask = np.nonzero(vocab_mask)[0]
X_source = X_source[:,vocab_mask]
X_target = X_target[:,vocab_mask]
vocab = [vocab[v] for v in vocab_mask]

## Generate Corpus
corpus, missing = generate_corpus(X_source, X_target, vocab, source=True, target=True)

## Initialize LDA Model
n_iter = 1000
n_burn = 250
model = tp.LDAModel(alpha=0.01,
                    eta=0.01,
                    k=50,
                    min_df=min_user_freq,
                    rm_top=250,
                    corpus=corpus,
                    seed=42)

## Initialize Sampler
model.train(1, workers=8)

## Corpus Parameters
V = model.num_vocabs
N = len(model.docs)
K = model.k

## Gibbs Cache
ll = np.zeros(n_iter)
phi = np.zeros((n_iter, K, V))
theta_train = np.zeros((n_iter, N, K))

## Train LDA Model
for epoch in tqdm(range(0, n_iter), desc="MCMC Iteration", file=sys.stdout):
    ## Run Sample Epoch
    model.train(1, workers=8)
    ## Examine Data Fit
    ll[epoch] = model.ll_per_word
    ## Cache Parameters
    phi[epoch] = np.vstack([model.get_topic_word_dist(i) for i in range(K)])
    epoch_theta_train = [model.infer(d,iter=100)[0] for d in model.docs]
    theta_train[epoch] = np.vstack(epoch_theta_train)

## Cache Model
_ = model.summary(topic_word_top_n=20, file=open(f"{output_dir}model_summary.txt","w"))

## Get Ground Truth Labels
y_train = np.array(
    [j for i, j in enumerate(y_source) if i not in missing.get("source")] + \
    [j for i, j in enumerate(y_target) if i not in missing.get("target")]
)

## Domain Indicies
source_train_ind = list(range(X_source.shape[0] - len(missing.get("source"))))
target_train_ind = list(range(len(source_train_ind), y_train.shape[0]))

## Separate Training Labels
y_train_s = y_train[source_train_ind]
y_train_t = y_train[target_train_ind]

## Isolate Feature Sets
theta_train_post_s = theta_train[n_burn:, source_train_ind, :]
theta_train_post_t = theta_train[n_burn:, target_train_ind, :]

## Average Topic Weights
s_avg_dist = theta_train_post_s.sum(axis=0).sum(axis=0) / (theta_train_post_s.shape[0] * theta_train_post_s.shape[1])
t_avg_dist = theta_train_post_t.sum(axis=0).sum(axis=0) / (theta_train_post_t.shape[0] * theta_train_post_t.shape[1])

## Train Classifiers and Cache Coefficients
coefs_source = np.zeros((theta_train_post_s.shape[0], theta_train_post_s.shape[2]))
coefs_target = np.zeros((theta_train_post_t.shape[0], theta_train_post_t.shape[2]))
for m, (x_source, x_target) in tqdm(enumerate(zip(theta_train_post_s, theta_train_post_t)), total=coefs_source.shape[0]):
    msource = LogisticRegression(); msource.fit(x_source, y_train_s)
    mtarget = LogisticRegression(); mtarget.fit(x_target, y_train_t)
    coefs_source[m] = msource.coef_[0]
    coefs_target[m] = mtarget.coef_[0]

## Compare Classifier Coefficients
q_coef_source = np.percentile(coefs_source, q=[2.5,50,97.5], axis=0)
q_coef_target = np.percentile(coefs_target, q=[2.5,50,97.5], axis=0)
fig, ax = plt.subplots(figsize=(10,5.8))
m = ax.scatter(q_coef_source[1],
               q_coef_target[1],
               c=s_avg_dist - t_avg_dist,
               cmap=plt.cm.coolwarm,
               s=75)
ax.errorbar(q_coef_source[1],
            q_coef_target[1],
            xerr=np.vstack([q_coef_source[1]-q_coef_source[0],q_coef_source[2]-q_coef_source[1]]),
            yerr=np.vstack([q_coef_target[1]-q_coef_target[0],q_coef_target[2]-q_coef_target[1]]),
            fmt="none",
            marker="none",
            zorder=0,
            markersize=20,
            alpha=0.5,
            label="Topic Dimension")
cbar = fig.colorbar(m)
cbar.set_label("Difference in Topic Prevalence\n({} - {})".format(DATASET_NAMES[source], DATASET_NAMES[target]),
               fontweight="bold",
               labelpad=10,
               fontsize=18)
cbar.ax.tick_params(labelsize=16) 
xlim = ax.get_xlim(); xlim = [-max(list(map(abs, xlim))), max(list(map(abs, xlim)))]
ylim = ax.get_ylim(); ylim = [-max(list(map(abs, ylim))), max(list(map(abs, ylim)))]
ax.fill_between([0, xlim[1]], [ylim[0], ylim[0]], [0, 0], color="gray", alpha=0.2, label="Domain Disagreement")
ax.fill_between([xlim[0], 0], [0, 0], [ylim[1], ylim[1]], color="gray", alpha=0.2)
ax.legend(loc="upper left", frameon=True, fontsize=14)
ax.axvline(0, color="black", linestyle="--", alpha=0.5)
ax.axhline(0, color="black", linestyle="--", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Linear Coefficient ({})".format(DATASET_NAMES[source]), fontweight="bold", fontsize=18)
ax.set_ylabel("Linear Coefficient ({})".format(DATASET_NAMES[target]), fontweight="bold", fontsize=18)
ax.tick_params(labelsize=16)
ax.set_xlim(xlim[0], xlim[1])
ax.set_ylim(ylim[0], ylim[1])
fig.tight_layout()
fig.savefig(f"{output_dir}topic_discriminator_coefficients.png",dpi=300)
fig.savefig(f"{output_dir}topic_discriminator_coefficients.pdf")
plt.close(fig)

## Compute Difference in Coefficients
q_coef_diff = np.percentile(coefs_source - coefs_target, q=[2.5,50,97.5], axis=0).T
q_coef_diff = pd.DataFrame(q_coef_diff, columns=["lower","median","upper"])
q_coef_diff["topic_reps"] = q_coef_diff.index.map(lambda k: ", ".join([i[0] for i in model.get_topic_words(k, 5)]))

## Visualize Differences
top_coef_diff = q_coef_diff["median"].nlargest(15).index.tolist() + q_coef_diff["median"].nsmallest(15).index.tolist()
top_coef_diff = q_coef_diff.loc[top_coef_diff].sort_values("median", ascending=True)
fig, ax = plt.subplots(figsize=(10,5.8))
ax.barh(range(top_coef_diff.shape[0]),
        left=top_coef_diff["lower"],
        width=top_coef_diff["upper"]-top_coef_diff["lower"],
        alpha=0.5,
        color="C0")
ax.scatter(top_coef_diff["median"],
           range(top_coef_diff.shape[0]),
           color="navy",
           alpha=0.8)
ax.axvline(0, color="black", linestyle="--", alpha=0.5)
ax.set_yticks(range(top_coef_diff.shape[0]))
ax.set_yticklabels(top_coef_diff["topic_reps"])
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("Coefficient Difference\n({} - {})".format(DATASET_NAMES[source], DATASET_NAMES[target]), fontweight="bold", fontsize=18)
ax.tick_params(axis="x",labelsize=16)
ax.tick_params(axis="y",labelsize=10)
ax.set_ylim(-.5, top_coef_diff.shape[0]-0.5)
fig.tight_layout()
fig.savefig(f"{output_dir}topic_discriminator_coefficients_differences.png",dpi=300)
fig.savefig(f"{output_dir}topic_discriminator_coefficients_differences.pdf")
plt.close(fig)

LOGGER.info("Script complete!")