
######################
### Imports
######################

## Standard Libary
import os
import sys
import json
import argparse

## External Libraries
import numpy as np
import pandas as pd
import tomotopy as tp
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn import metrics
from mhlib.util.helpers import flatten

######################
### Functions
######################

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
    parser.add_argument("--make_plots",
                        action="store_true",
                        default=False)
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file does not exist: {args.config}")
    return args

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

def standardize(X_latent,
                D):
    """

    """
    ## Standardize Latent X
    X_latent_normed = np.zeros_like(X_latent)
    X_latent_normed[D==0] = (X_latent[D==0] - X_latent[D==0].mean(axis=0)) / X_latent[D==0].std(axis=0)
    X_latent_normed[D==1] = (X_latent[D==1] - X_latent[D==1].mean(axis=0)) / X_latent[D==1].std(axis=0)
    return X_latent_normed

def data_generating_process(N,
                            sigma_0,
                            p_domain,
                            gamma,
                            V,
                            theta,
                            coef,
                            beta=None,
                            random_state=None):
    """

    """
    ## Set Random State
    if random_state is not None:
        np.random.seed(random_state)
    ## Update Beta
    if beta is None:
        beta = 1 / V
    ## Convert Data Types
    theta = np.array(theta)
    coef = np.array(coef)
    ## Generate Topic-Word Distributions
    phi = stats.dirichlet([beta]*V).rvs(theta.shape[1])
    ## Normalization of Parameters
    theta = theta / theta.sum(axis=1,keepdims=True)
    phi = phi / phi.sum(axis=1,keepdims=True)
    ## Update Document Topic Concentration
    theta = theta * sigma_0
    ## Data Storage
    X_latent = np.zeros((N,coef.shape[1]), dtype=float)
    X = np.zeros((N, phi.shape[1]), dtype=int)
    D = np.zeros(N, dtype=int)
    ## Sample Procedure
    for n in tqdm(range(N), "Sampling"):
        ## Sample Domain
        D[n] = int(np.random.rand() < p_domain)
        ## Sample Document Topic Mixture (Conditioned on Domain)
        X_latent[n] = stats.dirichlet(theta[D[n]]).rvs()
        ## Sample Number of Words
        n_d = stats.poisson(gamma).rvs()
        ## Create Document
        for _ in range(n_d):
            ## Sample Topic
            z = np.where(stats.multinomial(1, X_latent[n]).rvs()[0]>0)[0][0]
            ## Sample Word
            w = np.random.choice(phi.shape[1], p=phi[z])
            ## Cache
            X[n,w]+=1  
    ## Standardize
    X_latent_normed = standardize(X_latent, D)
    ## Compute P(y)
    py = np.zeros(N)
    py[D==0] = (1 / (1 + np.exp(-coef[[0]].dot(X_latent_normed[D==0].T))))[0]
    py[D==1] = (1 / (1 + np.exp(-coef[[1]].dot(X_latent_normed[D==1].T))))[0]
    ## Sample Y
    y = np.zeros(N)
    y[D==0] = (np.random.rand((D==0).sum()) < py[D==0]).astype(int)
    y[D==1] = (np.random.rand((D==1).sum()) < py[D==1]).astype(int)
    return X_latent, X, y, D, theta, phi


def fit_latent_regression(X_latent,
                          y,
                          D,
                          coef):
    """

    """
    ## Standardize Data
    X_latent_normed = standardize(X_latent, D)
    ## Fit Latent Models (All Variables)
    l0 = LogisticRegression(penalty="none"); l0.fit(X_latent_normed[D==0],y[D==0])
    l1 = LogisticRegression(penalty="none"); l1.fit(X_latent_normed[D==1],y[D==1])
    lall = np.vstack([l0.coef_, l1.coef_])
    ## Fit Latent Models (Individual Variables)
    lall_ind = np.zeros_like(lall)
    for d in [0,1]:
        for c in range(X_latent_normed.shape[1]):
            ldc = LogisticRegression(penalty="none")
            ldc.fit(X_latent_normed[D==d,c].reshape(-1,1), y[D==d])
            lall_ind[d, c] = ldc.coef_[0,0]
    ## Plot Marginals
    x = np.linspace(X_latent_normed.min(), X_latent_normed.max())
    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(10,5.8))
    for d, marker in zip([0,1],["o","x"]):
        dmask = np.where(D == d)[0]
        for i, c in enumerate(lall[d]):
            ax[i, d].scatter(X_latent_normed[dmask,i], y[dmask], marker=marker, color=f"C{d}", alpha=0.5)
            ax[i, d].plot(x, 1 / (1 + np.exp(-c * x)), color="black", alpha=0.5, label="Learned (Joint)")
            ax[i, d].plot(x, 1 / (1 + np.exp(-lall_ind[d,i] * x)), color="black", linestyle="--", alpha=0.5, label="Learned (Independent)")
            ax[i, d].plot(x, 1 / (1 + np.exp(-coef[d][i] * x)), color="green", alpha=0.5, label="Oracle")
            ax[i, d].set_xlabel(f"Feature {i}", fontweight="bold")
            ax[i, 0].set_ylabel("Outcome", fontweight="bold")
    for a in ax:
        for b in a:
            b.spines["right"].set_visible(False)
            b.spines["top"].set_visible(False)
            b.axvline(0, color="black", alpha=0.5, linestyle="--")
    for i, t in enumerate(["Source Domain","Target Domain"]):
        ax[0,i].set_title(t, fontweight="bold")
    ax[0,0].legend(loc="lower right")
    fig.tight_layout()
    return fig, ax

## Helper Function
doc_to_str = lambda x: flatten([[str(i)]*int(j) for i, j in enumerate(x)])

## Scoring
def score_model(y,
                y_test_lda,
                y_test_plda,
                D,
                test_ind,
                verbose=True):
    """

    """
    ## Score Cache
    scores = {"LDA":{},"PLDA":{}}
    ## Printing
    if verbose:
        print("~~~~~~ Test Set Performance ~~~~~~")
    ## Cycle through Domain Groups
    for d, domain in enumerate(["Source","Target","Overall"]):
        ## Domain Indices
        if d == 2:
            domain_test_ind = test_ind
        else:
            domain_test_ind = sorted(set(test_ind) & set(np.where(D==d)[0]))
        ## Compute Scores
        lda_f1 = metrics.f1_score(y[domain_test_ind], y_test_lda[domain_test_ind]>0.5)
        plda_f1 = metrics.f1_score(y[domain_test_ind], y_test_plda[domain_test_ind]>0.5)
        lda_auc = metrics.roc_auc_score(y[domain_test_ind], y_test_lda[domain_test_ind])
        plda_auc = metrics.roc_auc_score(y[domain_test_ind], y_test_plda[domain_test_ind])
        ## Cache
        scores["LDA"][domain] = {"f1":lda_f1, "auc":lda_auc}
        scores["PLDA"][domain] = {"f1":plda_f1, "auc":plda_auc}
        if verbose:
            print(domain,"Domain")
            print("LDA:  AUC={:.4f}, F1={:.4f}".format(lda_auc, lda_f1))
            print("PLDA: AUC={:.4f}, F1={:.4f}".format(plda_auc, plda_f1))
    return scores

def main():
    """

    """
    ######################
    ### Setup
    ######################
    ## Parse Command Line
    args = parse_arguments()
    ## Load Configuration
    config = Config(filepath=args.config)
    ## Output
    if config.output_dir is not None and not os.path.exists(config.output_dir):
        _ = os.makedirs(config.output_dir)
    ## Set Random State
    if config.random_state is not None:
        np.random.seed(config.random_state)
    ######################
    ### Data Generating Process
    ######################
    ## Generate Data
    X_latent, X, y, D, theta, phi = data_generating_process(config.N,
                                                            config.sigma_0,
                                                            config.p_domain,
                                                            config.gamma,
                                                            config.V,
                                                            config.theta,
                                                            config.coef,
                                                            beta=config.beta,
                                                            random_state=config.random_state)
    ## Data Distribution Plot
    if args.make_plots:
        fig, ax = fit_latent_regression(X_latent,
                                        y,
                                        D,
                                        config.coef)
        plt.show()
    ######################
    ### Fit Topic Models
    ######################
    ## Split Data into Training and Test
    train_ind = list(range(int(config.N*.8)))
    test_ind = list(range(int(config.N*.8),config.N))
    ## Initialize Models (3 Topics Total)
    lda = tp.LDAModel(k=3,
                      seed=config.random_state if config.random_state is not None else np.random.randint(1e6))
    plda = tp.PLDAModel(latent_topics=1,
                        topics_per_label=1,
                        seed=config.random_state if config.random_state is not None else np.random.randint(1e6))
    ## Add Training Data
    for n in train_ind:
        doc_n = doc_to_str(X[n])
        lda.add_doc(doc_n)
        plda.add_doc(doc_n, [str(D[n])])
    ## Initialize Sampler
    lda.train(1)
    plda.train(1)
    ## Update Parameters based on Corpus
    V_nn = lda.num_vocabs
    ## Generate Documents for Inference
    docs_lda = [lda.make_doc(doc_to_str(x)) for x in X]
    docs_plda = [plda.make_doc(doc_to_str(x), [str(D[i])]) for i, x in enumerate(X)]
    ## MCMC Storage
    n_iter = max(config.n_iter_lda, config.n_iter_plda)
    likelihood = np.zeros((n_iter, 2)) * np.nan
    theta_lda = np.zeros((n_iter, config.N, 3)) * np.nan
    theta_plda = np.zeros((n_iter, config.N, 3)) * np.nan
    phi_lda = np.zeros((n_iter, 3, V_nn)) * np.nan
    phi_plda = np.zeros((n_iter, 3, V_nn)) * np.nan
    ## Train LDA Model
    for epoch in tqdm(range(config.n_iter_lda), desc="LDA Training"):
        lda.train(1)
        likelihood[epoch,0] = lda.ll_per_word
        theta_lda[epoch] = np.vstack(lda.infer(docs_lda, iter=config.n_sample)[0])
        phi_lda[epoch] = np.vstack([lda.get_topic_word_dist(t) for t in range(lda.k)])
    ## Train PLDA Model
    for epoch in tqdm(range(config.n_iter_plda), desc="PLDA Training"):
        plda.train(1)
        likelihood[epoch,1] = plda.ll_per_word
        theta_plda[epoch] = np.vstack(plda.infer(docs_plda, iter=config.n_sample)[0])
        phi_plda[epoch] = np.vstack([plda.get_topic_word_dist(t) for t in range(plda.k)])
    ## Plot Likelihood
    if args.make_plots:
        plt.figure(figsize=(10,5.8))
        plt.plot(likelihood[:,0], label="LDA")
        plt.plot(likelihood[:,1], label="PLDA")
        plt.xlabel("Training Epoch", fontweight="bold")
        plt.ylabel("Log Likelihood Per Word", fontweight="bold")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    ## Plot Traces for Phi
    if args.make_plots:
        fig, axes = plt.subplots(phi_lda.shape[1], 2, figsize=(10,5.8))
        for m, (mphi,mdl) in enumerate(zip([phi_lda, phi_plda],["LDA","PLDA"])):
            ax = axes[:,m]
            for k in range(mphi.shape[1]):
                ax[k].plot(mphi[:,k,:])
                ax[k].set_ylabel("Parameter Value", fontweight="bold")
                ax[k].spines["top"].set_visible(False)
                ax[k].spines["right"].set_visible(False)
            ax[k].set_xlabel("Training Epoch", fontweight="bold")
            ax[0].set_title(f"{mdl} $\\phi$ Trace", fontweight="bold")
        fig.tight_layout()
        plt.show()
    ## Plot Sample Traces for Theta
    if args.make_plots:
        fig, ax = plt.subplots(5, 2, sharex=False, figsize=(10,5.8))
        for d, doc in enumerate(sorted(np.random.choice(config.N, 5, replace=False))):
            ax[d,0].plot(theta_lda[:,doc,:])
            ax[d,1].plot(theta_plda[:,doc,:])
            for i in range(2):
                ax[d,i].spines["right"].set_visible(False)
                ax[d,i].spines["top"].set_visible(False)
                ax[d,i].set_title(f"Document {doc}", loc="left", fontstyle="italic")
                ax[d,i].set_ylabel("$\\theta$")
        for m, mdl in enumerate(["LDA","PLDA"]):
            ax[-1, m].set_xlabel(f"{mdl} Training Epoch", fontweight="bold")
        fig.tight_layout()
        plt.show()
    ## Get Final Representations
    X_latent_lda = np.vstack(lda.infer(docs_lda, iter=config.n_sample, together=True)[0])
    X_latent_plda = np.vstack(plda.infer(docs_plda, iter=config.n_sample, together=True)[0])
    ## Isolate Latent Variables and Normalize
    X_latent_plda = X_latent_plda[:,-plda.latent_topics:]
    ## Fit Classifiers
    source_train_ind = sorted(set(train_ind) & set(np.where(D==0)[0]))
    lr_lda = LogisticRegression(); lr_lda.fit(X_latent_lda[source_train_ind], y[source_train_ind])
    lr_plda = LogisticRegression(); lr_plda.fit(X_latent_plda[source_train_ind], y[source_train_ind])
    ## Make Test Predictions
    y_test_lda = lr_lda.predict_proba(X_latent_lda)[:,1]
    y_test_plda = lr_plda.predict_proba(X_latent_plda)[:,1]
    ## Score Predictions
    scores = score_model(y, y_test_lda, y_test_plda, D, test_ind, True)
    if config.output_dir is not None and config.run_id is not None:
        with open(f"{config.output_dir}/{config.run_id}.scores.json","w") as the_file:
            json.dump(scores, the_file)

##################
### Execute
##################

if __name__ == "__main__":
    _ = main()