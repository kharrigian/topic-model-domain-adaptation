

"""
Train a topic model and associated classifier
for depression classification.
"""

#################
### Imports
#################

## Standard Library
import os
import sys
import json
import argparse
from multiprocessing import Pool

## External Libraries
import demoji
import numpy as np
import pandas as pd
from tqdm import tqdm
import tomotopy as tp
from scipy import sparse
import matplotlib.pyplot as plt
from mhlib.util.helpers import flatten
from mhlib.util.logging import initialize_logger
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

#################
### Globals
#################

## Logging
LOGGER = initialize_logger()

## Data Directories
DEPRESSION_DATA_DIR = f"./data/raw/depression/"

#################
### Helpers
#################

def parse_arguments():
    """

    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="")
    ## Required Arguments
    parser.add_argument("config",
                        type=str,
                        help="Path to configuration file")
    ## Optional Arguments
    parser.add_argument("--plot_document_topic",
                        action="store_true",
                        default=False)
    parser.add_argument("--plot_topic_word",
                        action="store_true",
                        default=False)
    parser.add_argument("--plot_fmt",
                        type=str,
                        default=".png")
    parser.add_argument("--evaluate_test",
                        action="store_true",
                        default=False)
    parser.add_argument("--fold",
                        type=int,
                        default=None)
    parser.add_argument("--k_folds",
                        type=int,
                        default=5)
    parser.add_argument("--num_jobs",
                        type=int,
                        default=8)
    parser.add_argument("--cache_parameters",
                        action="store_true",
                        default=False)
    parser.add_argument("--learn_threshold",
                        action="store_true",
                        default=False)
    parser.add_argument("--cache_predictions",
                        action="store_true",
                        default=False)
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file does not exist: {args.config}")
    if (args.plot_document_topic or args.plot_topic_word) and not args.cache_parameters:
        raise ValueError("Plotting requires parameter caching turned on.")
    return args

def replace_emojis(features):
    """
    
    """
    features_clean = []
    for f in features:
        f_res = demoji.findall(f)
        if len(f_res) > 0:
            for x,y in f_res.items():
                f = f.replace(x,f"<{y}>")
            features_clean.append(f)
        else:
            features_clean.append(f)
    return features_clean

class Config(object):

    """

    """

    def __init__(self, filepath):
        """

        """
        with open(filepath,"r") as the_file:
            config = json.load(the_file)
        for key, value in config.items():
            setattr(self, key, value)

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

def align_data(X_source,
               X_target,
               vocab_source,
               vocab_target,
               how="outer"):
    """

    """
    ## Make Vocabulary a Global Resource
    global vocab
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
    ## Format Emojis
    vocab = replace_emojis(vocab)
    return Xs.tocsr(), Xt.tocsr(), vocab

def split_data(X,
               y,
               splits):
    """

    """
    ## Masks
    train_ind = [i for i, s in enumerate(splits) if s == "train"]
    dev_ind = [i for i, s in enumerate(splits) if s == "dev"]
    test_ind = [i for i, s in enumerate(splits) if s == "test"]
    ## Get Data
    X_train, y_train = X[train_ind], y[train_ind]
    X_dev, y_dev = X[dev_ind], y[dev_ind]
    X_test, y_test = X[test_ind], y[dev_ind]
    return X_train, X_dev, X_test, y_train, y_dev, y_test

## Helper Function for Converting Count Data
term_expansion = lambda x, v: flatten([[v[i]] * int(x[0,i]) for i in x.nonzero()[1]])

def count_to_doc(x):
    """

    """
    ## Split Input
    i, x = x
    ## Check Size
    if x.getnnz() == 0:
        return (i, None)
    ## Expand and Return
    x_flat = term_expansion(x, vocab)
    return (i, x_flat)

def generate_corpus(X,
                    label,
                    mask=None,
                    corpus=None,
                    missing={},
                    num_jobs=8):
    """

    """
    ## Format Mask
    if mask is not None:
        mask = set(mask)
    ## Initialize Corpus (if Necessary)
    if corpus is None:
        corpus = tp.utils.Corpus()
    ## Initialize Missing Cache
    missing[label] = []
    ## Make Documents
    mp = Pool(num_jobs)
    docs = list(tqdm(mp.imap_unordered(count_to_doc, enumerate(X)),
                     total=X.shape[0],
                     desc="Creating Documents",
                     file=sys.stdout))
    docs = [d[1] for d in sorted(docs, key=lambda i: i[0])]
    _ = mp.close()
    ## Add Documents Incrementally
    for i, d in tqdm(enumerate(docs), total=len(docs), desc="Adding Documents", file=sys.stdout):
        if mask is not None and i not in mask:
            missing[label].append(i)
            continue
        if d is None:
            missing[label].append(i)
            continue
        corpus.add_doc(d,labels=[label])
    return corpus, missing

def which_plda_topic(topic_n,
                     plda_model):
    """

    """
    ## Model Information
    labels = plda_model.topic_label_dict
    topics_per_label = plda_model.topics_per_label
    ## Logical Deduction
    cur_lbl = 0
    cur_top_lbl = 0
    for i in range(topic_n):
        if (i + 1) % topics_per_label == 0:
            if cur_lbl < len(labels):
                cur_top_lbl = -1
            cur_lbl += 1
        cur_top_lbl += 1
    if cur_lbl >= len(labels):
        topic = "Latent Topic {} (#{} Overall)".format(cur_top_lbl, topic_n)
    else:
        topic = "{} Topic {} (#{} Overall)".format(labels[cur_lbl].title(), cur_top_lbl, topic_n)
    return topic

def plot_average_topic_distribution(theta,
                                    model,
                                    use_plda=False,
                                    n_burn=100):
    """

    """
    fig, ax = plt.subplots(figsize=(10,5.8))
    ax.imshow(theta[n_burn:].mean(axis=0), aspect="auto", cmap=plt.cm.Blues)
    if use_plda:
        for l, lbl in enumerate(model.topic_label_dict):
            ax.axvline((l+1) * model.topics_per_label - .5, alpha=0.5)
    ax.set_xlabel("Topic", fontweight="bold")
    ax.set_ylabel("Document", fontweight="bold")
    fig.tight_layout()
    return fig, ax
    
def plot_document_topic_distribution(doc,
                                     theta):
    """

    """
    ## Get Distribution
    doc_topic_quantile = np.percentile(theta[:,doc,:], q=[2.5,50,97.5], axis=0).T
    ## Plot Trace and Distribution
    fig, ax = plt.subplots(1,2,figsize=(10,5.6))
    for i in theta[:,doc,:].T:
        ax[0].plot(i, alpha=0.3)
    ax[1].bar(range(doc_topic_quantile.shape[0]),
            bottom=doc_topic_quantile[:,0],
            height=doc_topic_quantile[:,2]-doc_topic_quantile[:,0],
            color="C0",
            alpha=0.6)
    ax[1].scatter(range(doc_topic_quantile.shape[0]),
                doc_topic_quantile[:,1],
                color="navy",
                alpha=0.8,
                s=20)
    ax[0].set_ylabel("Parameter Sample", fontweight="bold")
    ax[0].set_xlabel("MCMC Iteration", fontweight="bold")
    ax[1].set_xlabel("Topic", fontweight="bold")
    ax[1].set_ylabel("Topic Proportion (Post Burn-in)", fontweight="bold")
    ax[0].set_ylim(0)
    ax[0].set_xlim(0,theta.shape[0])
    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.suptitle(f"Topic Distribution: Document {doc}", fontweight="bold", y=.97)
    fig.tight_layout()
    fig.subplots_adjust(top=.92)
    return fig, ax

def plot_topic_word_distribution(topic,
                                 model,
                                 phi,
                                 use_plda=False,
                                 n_trace=30,
                                 n_top=30,
                                 n_burn=100):
    """

    """
    ## Compute Quantile Range
    topic_quantile = pd.DataFrame(np.nanpercentile(phi[n_burn:,topic,:], q=[2.5,50,97.5], axis=0).T,
                                  index=model.used_vocabs,
                                  columns=["lower","median","upper"])
    top_topic_quantile = topic_quantile.loc[topic_quantile["median"].nlargest(n_top).index].iloc[::-1]
    ## Generate Plot
    fig, ax = plt.subplots(1,2,figsize=(10,5.8))
    for term_trace in phi[:,topic,:].T[np.argsort(topic_quantile["median"].values)[-n_trace:]]:
        ax[0].plot(term_trace, alpha=0.3)
    ax[1].barh(range(n_top),
            left=top_topic_quantile["lower"],
            width=top_topic_quantile["upper"]-top_topic_quantile["lower"],
            color="C0",
            alpha=0.6)
    ax[1].scatter(top_topic_quantile["median"],
                range(n_top),
                color="C0",
                alpha=.8)
    ax[1].set_yticks(range(n_top))
    ax[1].set_yticklabels(top_topic_quantile.index.tolist())
    ax[1].set_ylim(-.5, n_top-.5)
    ax[1].set_xlabel("Parameter Range (Post Burn-in)", fontweight="bold")
    ax[1].set_ylabel("Vocabulary Term", fontweight="bold")
    ax[0].set_xlim(0,phi.shape[0])
    ax[0].set_ylim(0)
    ax[0].set_xlabel("MCMC Iteration", fontweight="bold")
    ax[0].set_ylabel("Parameter Sample", fontweight="bold")
    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    if use_plda:
        fig.suptitle("Term Distribution for {}".format(which_plda_topic(topic, model)), fontweight="bold", y=.975)
    else:
        fig.suptitle("Term Distribution for Topic {}".format(topic), fontweight="bold", y=.975)
    fig.subplots_adjust(top=.92)
    return fig, ax

def get_scores(y_true,
               y_score,
               threshold=0.5):
    """

    """
    ## Binarize
    y_pred_bin = (y_score > threshold).astype(int)
    ## Score
    tpr, fpr, thresh = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(tpr, fpr)
    avg_precision = metrics.average_precision_score(y_true, y_score)
    f1 = metrics.f1_score(y_true, y_pred_bin)
    precision = metrics.precision_score(y_true, y_pred_bin)
    recall = metrics.recall_score(y_true, y_pred_bin)
    return tpr, fpr, {"auc":auc,"avg_precision":avg_precision,"f1":f1,"precision":precision,"recall":recall}

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
    ## Check Size
    if sample_size is not None and sample_size > X.shape[0]:
        LOGGER.warning("Requested sample size ({}) greater than available. Reducing to max ({}).".format(sample_size, X.shape[0]))
        sample_size = X.shape[0]
    ## Rebalance Data
    X, y = _rebalance(X, y, class_ratio, random_seed)
    ## Downsample Data
    X, y = _downsample(X, y, sample_size, random_seed)
    return X, y

def valid_sampler(config):
    """

    """
    for epoch in range(0, config.n_iter):
        if (epoch + 1) >= config.n_burn and (epoch + 1) % config.infer_sample_rate == 0:
            return True
    return False

def main():
    """

    """
    ###################
    ### Script Setup
    ###################
    ## Parse Command Line
    args = parse_arguments()
    ## Load Configuration
    config = Config(filepath=args.config)
    ## Check Sampler
    if not valid_sampler(config):
        raise ValueError("Configuration results in no inferences. Change burn-in or sample frequency.")
    ## Create Output Directories
    basedir = f"{config.output_dir}/" if args.fold is None else f"{config.output_dir}/fold-{args.fold}/".replace("//","/")
    dirs = ["topic_model/document_topic/","topic_model/topic_word/","classification/"]
    for d in dirs:
        ddir = f"{basedir}{d}"
        if not os.path.exists(ddir):
            _ = os.makedirs(ddir)
    ## Cache Configuration
    _ = os.system(f"cp {args.config} {config.output_dir}/config.json")
    ## Set Random Seed
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
    ###################
    ### Data Preparation
    ###################
    ## Load Data
    LOGGER.info("Loading Processed Datasets")
    X_source, y_source, splits_source, filenames_source, users_source, terms_source = load_data(f"{DEPRESSION_DATA_DIR}{config.source}/")
    X_target, y_target, splits_target, filenames_target, users_target, terms_target = load_data(f"{DEPRESSION_DATA_DIR}{config.target}/")
    ## Align Vocabulary Spaces
    LOGGER.info("Aligning Vocabularies")
    X_source, X_target, vocab = align_data(X_source, X_target, terms_source, terms_target, config.vocab_alignment)
    ## Split Data
    LOGGER.info("Separating Datasets by Split")
    Xs_train, Xs_dev, Xs_test, ys_train, ys_dev, ys_test = split_data(X_source, y_source, splits_source)
    Xt_train, Xt_dev, Xt_test, yt_train, yt_dev, yt_test = split_data(X_target, y_target, splits_target)
    ## Sampling 
    LOGGER.info("Sampling Source Data")
    Xs_train, ys_train = sample_data(Xs_train, ys_train, config.source_class_ratio.get("train"), config.source_sample_size.get("train"))
    Xs_dev, ys_dev = sample_data(Xs_dev, ys_dev, config.source_class_ratio.get("dev"), config.source_sample_size.get("dev"))
    Xs_test, ys_test = sample_data(Xs_test, ys_test, config.source_class_ratio.get("test"), config.source_sample_size.get("test"))
    LOGGER.info("Sampling Target Data")
    Xt_train, yt_train = sample_data(Xt_train, yt_train, config.target_class_ratio.get("train"), config.target_sample_size.get("train"))
    Xt_dev, yt_dev = sample_data(Xt_dev, yt_dev, config.target_class_ratio.get("dev"), config.target_sample_size.get("dev"))
    Xt_test, yt_test = sample_data(Xt_test, yt_test, config.target_class_ratio.get("test"), config.target_sample_size.get("test"))
    ## Cross Validation
    if args.fold is not None:
        LOGGER.info(f"Isolating K-Fold Data (Fold {args.fold})")
        ## Initialize Splitter
        splitter = StratifiedKFold(n_splits=args.k_folds,
                                   shuffle=True,
                                   random_state=config.random_seed)
        ## Merge Data
        Xs_all = sparse.vstack([Xs_train, Xs_dev])
        Xt_all = sparse.vstack([Xt_train, Xt_dev])
        ys_all = np.hstack([ys_train,ys_dev])
        yt_all = np.hstack([yt_train,yt_dev])
        ## Get Train and Dev Splits for the Fold
        splits_source = list(splitter.split(Xs_all, ys_all))[args.fold-1]
        splits_target = list(splitter.split(Xt_all, yt_all))[args.fold-1]
        ## Isolate Relevant Data
        Xs_train, ys_train = Xs_all[splits_source[0]], ys_all[splits_source[0]]
        Xs_dev, ys_dev = Xs_all[splits_source[1]], ys_all[splits_source[1]]
        Xt_train, yt_train = Xt_all[splits_target[0]], yt_all[splits_target[0]]
        Xt_dev, yt_dev = Xt_all[splits_target[1]], yt_all[splits_target[1]]
    ###################
    ### Corpus Generation
    ###################
    ## Sample Topic Model Training Masks
    if config.topic_model_data.get("source") is not None:
        if config.topic_model_data.get("source") > Xs_train.shape[0]:
            LOGGER.warning("Requested Source Topic Model Train Size Greater than Available Data. Downsizing.")
            config.topic_model_data["source"] = Xs_train.shape[0]
        source_mask = sorted(np.random.choice(Xs_train.shape[0], size=config.topic_model_data.get("source"), replace=False))
    else:
        source_mask = list(range(Xs_train.shape[0]))
    if config.topic_model_data.get("target") is not None:
        if config.topic_model_data.get("target") > Xt_train.shape[0]:
            LOGGER.warning("Requested Target Topic Model Train Size Greater than Available Data. Downsizing.")
            config.topic_model_data["target"] = Xt_train.shape[0]
        target_mask = sorted(np.random.choice(Xt_train.shape[0], size=config.topic_model_data.get("target"), replace=False))
    else:
        target_mask = list(range(Xt_train.shape[0]))
    ## Initialize Corpus
    LOGGER.info("Generating Training Corpus (Topic-Model Learning)")
    train_corpus, train_missing = generate_corpus(Xs_train, label="source", mask=source_mask, num_jobs=args.num_jobs)
    train_corpus, train_missing = generate_corpus(Xt_train, label="target", mask=target_mask, corpus=train_corpus, missing=train_missing, num_jobs=args.num_jobs)
    LOGGER.info("Generating Training Corpus (Inference)")
    if config.topic_model_data.get("source") is None and config.topic_model_data.get("target") is None:
        LOGGER.info("Using Training Corpus for Inference")
        train_corpus_infer = train_corpus
        train_missing_infer = train_missing
    else:
        train_corpus_infer, train_missing_infer = generate_corpus(Xs_train, label="source", missing={}, num_jobs=args.num_jobs)
        train_corpus_infer, train_missing_infer = generate_corpus(Xt_train, label="target", corpus=train_corpus_infer, missing=train_missing_infer, num_jobs=args.num_jobs)
    LOGGER.info("Generating Development Corpus (Inference)")
    development_corpus, dev_missing = generate_corpus(Xs_dev, label="source", missing={}, num_jobs=args.num_jobs)
    development_corpus, dev_missing = generate_corpus(Xt_dev, label="target", corpus=development_corpus, missing=dev_missing, num_jobs=args.num_jobs)
    if args.evaluate_test:
        LOGGER.info("Generating Test Corpus (Inference)")
        test_corpus, test_missing = generate_corpus(Xs_test, label="source", missing={}, num_jobs=args.num_jobs)
        test_corpus, test_missing = generate_corpus(Xt_test, label="target", corpus=test_corpus, missing=test_missing, num_jobs=args.num_jobs)
    ###################
    ### Topic Model (Training)
    ###################
    ## Initialize Model
    if config.use_plda:
        model = tp.PLDAModel(alpha=config.alpha,
                             eta=config.beta,
                             latent_topics=config.k_latent,
                             topics_per_label=config.k_per_label,
                             min_df=config.min_doc_freq,
                             rm_top=config.rm_top,
                             corpus=train_corpus,
                             seed=config.random_seed)
    else:
        model = tp.LDAModel(alpha=config.alpha,
                            eta=config.beta,
                            k=config.k_latent,
                            min_df=config.min_doc_freq,
                            rm_top=config.rm_top,
                            corpus=train_corpus,
                            seed=config.random_seed)
    ## Initialize Sampler
    model.train(1, workers=args.num_jobs)
    ## Corpus-Updated Parameters
    V = model.num_vocabs
    N = len(model.docs)
    N_train = len(train_corpus_infer)
    N_dev = len(development_corpus)
    N_test = len(test_corpus) if args.evaluate_test else 0
    K = model.k
    ## Gibbs Cache
    ll = np.zeros(config.n_iter)
    theta_train = []
    theta_dev = []
    theta_test = [] if args.evaluate_test else None
    if args.cache_parameters:
        phi = np.zeros((config.n_iter, K, V))
        theta = np.zeros((config.n_iter, N, K))
    else:
        phi = np.zeros((K,V))
        theta = np.zeros((N, K))
    ## Train Model
    for epoch in tqdm(range(0, config.n_iter), desc="MCMC Iteration", file=sys.stdout):
        ## Run Sample Epoch
        model.train(1, workers=args.num_jobs)
        ## Examine Data Fit
        ll[epoch] = model.ll_per_word
        ## Cache Model Parameters
        if args.cache_parameters:
            phi[epoch] = np.vstack([model.get_topic_word_dist(i) for i in range(K)])
            theta[epoch] = np.vstack([d.get_topic_dist() for d in model.docs])
        elif epoch == (config.n_iter - 1):
            phi = np.vstack([model.get_topic_word_dist(i) for i in range(K)])
            theta = np.vstack([d.get_topic_dist() for d in model.docs])
        ## Make Inferences Regularly
        if (epoch + 1) >= config.n_burn and (epoch + 1) % config.infer_sample_rate == 0:
            ## Training Inference
            train_dist, _ = model.infer(train_corpus_infer, iter=config.n_sample, together=False)
            theta_train.append(np.vstack([t.get_topic_dist() for t in train_dist]))
            ## Development Inference
            dev_dist, _ = model.infer(development_corpus, iter=config.n_sample, together=False)
            theta_dev.append(np.vstack([d.get_topic_dist() for d in dev_dist]))
            ## Test Inference
            if args.evaluate_test:
                test_dist, _ = model.infer(test_corpus, iter=config.n_sample, together=False)
                theta_test.append(np.vstack([t.get_topic_dist() for t in test_dist]))
    ## Stack Inferences
    theta_train = np.stack(theta_train)
    theta_dev = np.stack(theta_dev)
    if args.evaluate_test:
        theta_test = np.stack(theta_test)
    ## Cache Model Summary
    _ = model.summary(topic_word_top_n=20, file=open(f"{basedir}/topic_model/model_summary.txt","w"))
    ################
    ### Topic Model Diagnostics
    ################
    ## Plot Likelihood
    fig, ax = plt.subplots(figsize=(10,5.8))
    ax.plot(ll)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("MCMC Iteration", fontweight="bold")
    ax.set_ylabel("Log-Likelihood Per Word", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{basedir}/topic_model/log_likelihood_train{args.plot_fmt}",dpi=300)
    plt.close(fig)
    ## Evaluate Topics
    for k in range(model.k):
        top_terms = [i[0] for i in model.get_topic_words(k, top_n=20)]
        if config.use_plda:
            LOGGER.info("{}: {}".format(which_plda_topic(k, model), ", ".join(top_terms)))
        else:
            LOGGER.info("{}: {}".format(k, ", ".join(top_terms)))
    ## Show Average Topic Distribution (Training Data)
    LOGGER.info("Plotting Average Topic Distributions")
    try:
        fig, ax = plot_average_topic_distribution(theta=theta_train,
                                                  model=model,
                                                  use_plda=config.use_plda,
                                                  n_burn=0)
        fig.savefig(f"{basedir}/topic_model/average_topic_distribution_train{args.plot_fmt}",dpi=300)
        plt.close(fig)
    except:
        pass
    ## Show Average Topic Distribution (Development Data)
    try:
        fig, ax = plot_average_topic_distribution(theta=theta_dev,
                                                  model=model,
                                                  use_plda=config.use_plda,
                                                  n_burn=0)
        fig.savefig(f"{basedir}/topic_model/average_topic_distribution_development{args.plot_fmt}",dpi=300)
        plt.close(fig)
    except:
        pass
    ## Show Trace for a Document Topic Distribution (Random Sample)
    if args.plot_document_topic:
        LOGGER.info("Plotting Sample of Document Topic Distributions")
        for doc_n in np.random.choice(theta_train.shape[1], 10):
            try:
                fig, ax = plot_document_topic_distribution(doc=doc_n,
                                                           theta=theta)
                fig.savefig(f"{basedir}/topic_model/document_topic/train_{doc_n}{args.plot_fmt}",dpi=300)
                plt.close(fig)
            except:
                pass
    ## Show Trace for a Topic Word Distribution
    if args.plot_topic_word:
        LOGGER.info("Plotting Topic Word Distributions")
        for topic in tqdm(range(K), total=K, desc="Topic Word Distribution", file=sys.stdout):
            try:
                fig, ax = plot_topic_word_distribution(topic=topic,
                                                       phi=phi,
                                                       model=model,
                                                       use_plda=config.use_plda,
                                                       n_trace=30,
                                                       n_top=30,
                                                       n_burn=config.n_burn if config.n_burn < phi.shape[0] else -100)
                fig.savefig(f"{basedir}/topic_model/topic_word/topic_{topic}{args.plot_fmt}",dpi=300)
                plt.close(fig)
            except:
                pass
    ################
    ### Depression Classifier Training
    ################
    LOGGER.info("Beginning Classifier Training Procedure")
    ## Isolate General Latent Representations
    theta_train_latent = theta_train[:,:,-config.k_latent:]
    theta_dev_latent = theta_dev[:,:,-config.k_latent:]
    if args.evaluate_test:
        theta_test_latent = theta_test[:,:,-config.k_latent:]
    ## Get Ground Truth Labels
    y_train = np.array(
        [j for i, j in enumerate(ys_train) if i not in train_missing_infer.get("source")] + \
        [j for i, j in enumerate(yt_train) if i not in train_missing_infer.get("target")]
    )
    y_dev = np.array(
        [j for i, j in enumerate(ys_dev) if i not in dev_missing.get("source")] + \
        [j for i, j in enumerate(yt_dev) if i not in dev_missing.get("target")]
    )
    if args.evaluate_test:
        y_test = np.array(
        [j for i, j in enumerate(ys_test) if i not in test_missing.get("source")] + \
        [j for i, j in enumerate(yt_test) if i not in test_missing.get("target")]
    )
    ## Domain Masks
    source_train_ind = list(range(Xs_train.shape[0] - len(train_missing_infer.get("source"))))
    target_train_ind = list(range(len(source_train_ind), y_train.shape[0]))
    source_dev_ind = list(range(Xs_dev.shape[0] - len(dev_missing.get("source"))))
    target_dev_ind = list(range(len(source_dev_ind), y_dev.shape[0]))
    if args.evaluate_test:
        source_test_ind = list(range(Xs_test.shape[0] - len(test_missing.get("source"))))
        target_test_ind = list(range(len(source_test_ind), y_test.shape[0]))
    ## Separate Training Labels
    y_train_s = y_train[source_train_ind]
    y_train_t = y_train[target_train_ind]
    y_dev_s = y_dev[source_dev_ind]
    y_dev_t = y_dev[target_dev_ind]
    if args.evaluate_test:
        y_test_s = y_test[source_test_ind]
        y_test_t = y_test[target_test_ind]
    ## Caching
    if args.cache_predictions:
        ## Labels
        _ = np.save(f"{basedir}/classification/labels.train.npy", y_train)
        _ = np.save(f"{basedir}/classification/labels.dev.npy", y_dev)
        ## Indices
        _ = np.save(f"{basedir}/classification/source.train.npy", source_train_ind)
        _ = np.save(f"{basedir}/classification/target.train.npy", target_train_ind)
        _ = np.save(f"{basedir}/classification/source.dev.npy", source_dev_ind)
        _ = np.save(f"{basedir}/classification/target.dev.npy", target_dev_ind)
        if args.evaluate_test:
            ## Labels
            _ = np.save(f"{basedir}/classification/labels.test.npy", y_test)
            ## Indices
            _ = np.save(f"{basedir}/classification/source.test.npy", source_test_ind)
            _ = np.save(f"{basedir}/classification/target.test.npy", target_test_ind)
    ## Cycle Through Types of Preprocessing, Training, and Inference
    all_scores = []
    for C in config.C:
        for average_representation in config.averaging:
            for norm in config.norm:
                LOGGER.info("Feature Set: Average Representation ({}), Norm ({}), Regularization ({})".format(average_representation, norm, C))
                if average_representation:
                    ## Average
                    X_train = theta_train_latent.mean(axis=0)
                    X_dev = theta_dev_latent.mean(axis=0)
                    if args.evaluate_test:
                        X_test = theta_test_latent.mean(axis=0)
                    ## Normalization (If Desired)
                    if norm:
                        X_train = normalize(X_train, norm=norm, axis=1)
                        X_dev = normalize(X_dev, norm=norm, axis=1)
                        if args.evaluate_test:
                            X_test = normalize(X_test, norm=norm, axis=1)
                    ## Reshape Data
                    X_train = X_train.reshape((1,X_train.shape[0],X_train.shape[1]))
                    X_dev = X_dev.reshape((1,X_dev.shape[0], X_dev.shape[1]))
                    if args.evaluate_test:
                        X_test =  X_test.reshape((1,X_test.shape[0], X_test.shape[1]))
                else:
                    ## Remove Burn In
                    X_train = theta_train_latent.copy()
                    X_dev = theta_dev_latent.copy()
                    if args.evaluate_test:
                        X_test = theta_test_latent.copy()
                    ## Normalization (If Desired)
                    if norm:
                        X_train = np.stack([normalize(x, norm=norm, axis=1) for x in X_train])
                        X_dev = np.stack([normalize(x, norm=norm, axis=1) for x in X_dev])
                        if args.evaluate_test:
                            X_test = np.stack([normalize(x, norm=norm, axis=1) for x in X_test])
                ## Training
                models = []
                for x in tqdm(X_train, desc="Fitting Models", file=sys.stdout):
                    ## Fit Classifier
                    logit = LogisticRegression(C=C,
                                              random_state=42,
                                              max_iter=config.max_iter,
                                              solver='lbfgs')
                    logit.fit(x[source_train_ind], y_train[source_train_ind])
                    ## Get Predictions
                    models.append(logit)
                ## Inference
                y_pred_train = np.zeros((len(models), X_train.shape[0], y_train.shape[0]))
                y_pred_dev = np.zeros((len(models), X_dev.shape[0], y_dev.shape[0]))
                if args.evaluate_test:
                    y_pred_test = np.zeros((len(models), X_test.shape[0], y_test.shape[0]))
                for m, mdl in tqdm(enumerate(models), position=0, desc="Making Predictions", total=len(models), file=sys.stdout):
                    y_pred_train[m] = mdl.predict_proba(X_train[m])[:,1]
                    y_pred_dev[m] = mdl.predict_proba(X_dev[m])[:,1]
                    if args.evaluate_test:
                        y_pred_test[m] = mdl.predict_proba(X_test[m])[:,1]
                ## Cache Predictions
                if args.cache_predictions:
                    ## Predictions
                    _ = np.save(f"{basedir}/classification/predictions.train.{C}.{average_representation}.{norm}.npy", y_pred_train)
                    _ = np.save(f"{basedir}/classification/predictions.dev.{C}.{average_representation}.{norm}.npy", y_pred_dev)
                    if args.evaluate_test:
                        _ = np.save(f"{basedir}/classification/predictions.{C}.{average_representation}.{norm}.dev.npy", y_pred_test)
                ## Learn Optimal Thresholds (Youden's J-Score)
                thresholds = {}
                for m, mdl_pred in tqdm(enumerate(y_pred_dev), total=y_pred_dev.shape[0], desc="Learning Binarization Thresholds", file=sys.stdout):
                    for l, latent_pred in enumerate(mdl_pred):
                        for d, dind in enumerate([source_dev_ind, target_dev_ind]):
                            if args.learn_threshold:
                                d_l_pred = latent_pred[dind]
                                d_l_true = y_dev[dind]
                                if len(d_l_pred) == 0:
                                    continue
                                fpr, tpr, t = metrics.roc_curve(d_l_true, d_l_pred, drop_intermediate=False)
                                j_scores = tpr - fpr
                                j_ordered = sorted(zip(j_scores, t))
                                j_opt_thresh = j_ordered[-1][1]
                                thresholds[(m,l,d)] = j_opt_thresh
                            else:
                                thresholds[(m,l,d)] = 0.5
                ## ROC Curves
                LOGGER.info("Plotting ROC/AUC and Scoring Training/Development Predictions")
                auc_scores = [[[],[]],[[],[]]]
                fig, ax = plt.subplots(2, 2, figsize=(10,5.8), sharex=True, sharey=True)
                for m, mdl_pred in tqdm(enumerate(y_pred_train), total=y_pred_train.shape[0], desc="Train Scoring", file=sys.stdout):
                    for l, latent_pred in enumerate(mdl_pred):
                        for d, dind in enumerate([source_train_ind, target_train_ind]):
                            d_l_pred = latent_pred[dind]
                            d_l_true = y_train[dind]
                            if len(d_l_pred) == 0:
                                continue
                            tpr, fpr, dl_scores = get_scores(d_l_true, d_l_pred, threshold=thresholds[(m,l,d)])
                            auc_scores[d][0].append(dl_scores.get("auc",0))
                            dl_scores.update({"model_n":m,"domain":"source" if d == 0 else "target","group":"train","threshold":thresholds[(m,l,d)]})
                            dl_scores.update({"norm":norm, "is_average_representation":average_representation, "C":C})
                            all_scores.append(dl_scores)
                            ax[d][0].plot(tpr, fpr, alpha=0.01 if not average_representation else .8, color=f"navy", linewidth=0.5 if not average_representation else 1)
                for m, mdl_pred in tqdm(enumerate(y_pred_dev), total=y_pred_dev.shape[0], desc="Development Scoring", file=sys.stdout):
                    for l, latent_pred in enumerate(mdl_pred):
                        for d, dind in enumerate([source_dev_ind, target_dev_ind]):
                            d_l_pred = latent_pred[dind]
                            d_l_true = y_dev[dind]
                            if len(d_l_pred) == 0:
                                continue
                            tpr, fpr, dl_scores = get_scores(d_l_true, d_l_pred, threshold=thresholds[(m,l,d)])
                            dl_scores.update({"model_n":m,"domain":"source" if d == 0 else "target","group":"development","threshold":thresholds[(m,l,d)]})
                            dl_scores.update({"norm":norm, "is_average_representation":average_representation,"C":C})
                            all_scores.append(dl_scores)
                            auc_scores[d][1].append(dl_scores.get("auc",0))
                            ax[d][1].plot(tpr, fpr, alpha=0.01 if not average_representation else .8, color=f"navy", linewidth=0.5 if not average_representation else 1)
                if args.evaluate_test:
                    for m, mdl_pred in tqdm(enumerate(y_pred_test), total=y_pred_test.shape[0], desc="Test Scoring", file=sys.stdout):
                        for l, latent_pred in enumerate(mdl_pred):
                            for d, dind in enumerate([source_test_ind, target_test_ind]):
                                d_l_pred = latent_pred[dind]
                                d_l_true = y_test[dind]
                                if len(d_l_pred) == 0:
                                    continue
                                tpr, fpr, dl_scores = get_scores(d_l_true, d_l_pred, threshold=thresholds[(m,l,d)])
                                dl_scores.update({"model_n":m,"domain":"source" if d == 0 else "target","group":"test","threshold":thresholds[(m,l,d)]})
                                dl_scores.update({"norm":norm, "is_average_representation":average_representation,"C":C})
                                all_scores.append(dl_scores)
                for i, domain in enumerate(["Source","Target"]):
                    ax[-1,i].set_xlabel("True Positive Rate", fontweight="bold")
                    ax[i, 0].set_ylabel("False Positive Rate", fontweight="bold")
                    for j, group in enumerate(["Train","Development"]):
                        ax[i,j].plot([0,1],[0,1],color="black",linestyle="--")
                        ax[i,j].spines["top"].set_visible(False)
                        ax[i,j].spines["right"].set_visible(False)
                        ax[i,j].set_title(f"{domain} {group}", fontweight="bold")
                        if len(auc_scores[i][j]) > 0:
                            ax[i,j].plot([],[],color="navy",label="Mean AUC: {:.3f}".format(np.mean(auc_scores[i][j])))
                            ax[i,j].legend(loc="lower right")
                        ax[i,j].set_xlim(0,1)
                        ax[i,j].set_ylim(0,1)
                fig.tight_layout()
                fig.savefig(f"{basedir}/classification/roc_auc_{average_representation}_{norm}_{C}{args.plot_fmt}",dpi=300)
                plt.close(fig)
    ## Format Scores
    LOGGER.info("Caching Scores")
    all_scores_df = pd.DataFrame(all_scores)
    if args.fold is not None:
        all_scores_df["fold"] = args.fold
    all_scores_df.to_csv(f"{basedir}/classification/scores.csv",index=False)
    ## Script Complete
    LOGGER.info("Done!")

#################
### Execute
#################

## Run Program
if __name__ == "__main__":
    _ = main()