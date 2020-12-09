
######################
### Imports
######################

## Standard Libary
import os
import sys

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
### Data Generating Process
######################

## Data Parameters
N = 1000
sigma_0 = 3
p_domain = 0.5
gamma = 20

## Relative Concentrations
theta = np.array([[0.7, 0.2, 0.1],
                  [0.2, 0.7, 0.1]])
phi = np.array([[10, 5, 1, 1, 2, 2],
                [1, 1, 10, 5, 2, 3],
                [5, 15, 5, 1, 4, 7]])

## Probability of Target Within Each Domain
coef = np.array([[1, 0.05, 0.1],
                 [0.05, 1, 0.1]])

## Normalization of Parameters
theta = theta / theta.sum(axis=1,keepdims=True)
phi = phi / phi.sum(axis=1,keepdims=True)

## Update Concentration
theta = theta * sigma_0

## Data Storage
X_latent = np.zeros((N,coef.shape[1]), dtype=float)
X = np.zeros((N, phi.shape[1]), dtype=int)
D = np.zeros(N, dtype=int)

## Sample Procedure
for n in range(N):
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
    
## Standardize Latent X
X_latent_normed = np.zeros_like(X_latent)
X_latent_normed[D==0] = (X_latent[D==0] - X_latent[D==0].mean(axis=0)) / X_latent[D==0].std(axis=0)
X_latent_normed[D==1] = (X_latent[D==1] - X_latent[D==1].mean(axis=0)) / X_latent[D==1].std(axis=0)

## Compute P(y)
py = np.zeros(N)
py[D==0] = (1 / (1 + np.exp(-coef[[0]].dot(X_latent_normed[D==0].T))))[0]
py[D==1] = (1 / (1 + np.exp(-coef[[1]].dot(X_latent_normed[D==1].T))))[0]

## Sample Y
y = np.zeros(N)
y[D==0] = (np.random.rand((D==0).sum()) < py[D==0]).astype(int)
y[D==1] = (np.random.rand((D==1).sum()) < py[D==1]).astype(int)

## Fit Latent Models (All Variables)
l0 = LogisticRegression(penalty="none"); l0.fit(X_latent_normed[D==0],y[D==0])
l1 = LogisticRegression(penalty="none"); l1.fit(X_latent_normed[D==1],y[D==1])
lall = np.vstack([l0.coef_, l1.coef_])

## Plot Marginals
x = np.linspace(X_latent_normed.min(), X_latent_normed.max())
fig, ax = plt.subplots(3, 2, sharex=True, figsize=(10,5.8))
for d, marker in zip([0,1],["o","x"]):
    dmask = np.where(D == d)[0]
    for i, c in enumerate(lall[d]):
        ax[i, d].scatter(X_latent_normed[dmask,i], y[dmask], marker=marker, color=f"C{d}", alpha=0.5)
        ax[i, d].plot(x, 1 / (1 + np.exp(-c * x)), color="black", alpha=0.5)
        ax[i, d].plot(x, 1 / (1 + np.exp(-coef[d,i] * x)), color="green", alpha=0.5)
        ax[i, d].set_xlabel(f"Feature {i}", fontweight="bold")
        ax[i, 0].set_ylabel("Outcome", fontweight="bold")
for a in ax:
    for b in a:
        b.spines["right"].set_visible(False)
        b.spines["top"].set_visible(False)
        b.axvline(0, color="black", alpha=0.5, linestyle="--")
for i, t in enumerate(["Source Domain","Target Domain"]):
    ax[0,i].set_title(t, fontweight="bold")
fig.tight_layout()
plt.show()

######################
### Fit Topic Models
######################

## Helper Function
doc_to_str = lambda x: flatten([[str(i)]*int(j) for i, j in enumerate(x)])

## Split Data
train_ind = list(range(int(N*.8)))
test_ind = list(range(int(N*.8),N))

## Initialize Models (3 Topics Total)
n_iter = 500
lda = tp.LDAModel(k=3,
                  seed=42)
plda = tp.PLDAModel(latent_topics=1,
                    topics_per_label=1,
                    seed=42)

## Add Training Data
for n in train_ind:
    doc_n = doc_to_str(X[n])
    lda.add_doc(doc_n)
    plda.add_doc(doc_n, [str(D[n])])

## Generate Documents
docs_lda = [lda.make_doc(doc_to_str(x)) for x in X]
docs_plda = [plda.make_doc(doc_to_str(x), [str(D[i])]) for i, x in enumerate(X)]

## MCMC Storage
likelihood = np.zeros((n_iter, 2)) * np.nan
theta_lda = np.zeros((n_iter, N, 3)) * np.nan
phi_lda = np.zeros((n_iter, 3, 6)) * np.nan
theta_plda = np.zeros((n_iter, N, 3)) * np.nan
phi_plda = np.zeros((n_iter, 3, 6)) * np.nan

## Train LDA Model
for epoch in tqdm(range(100), desc="LDA Training"):
    lda.train(1)
    likelihood[epoch,0] = lda.ll_per_word
    theta_lda[epoch] = np.vstack(lda.infer(docs_lda, iter=100)[0])
    phi_lda[epoch] = np.vstack([lda.get_topic_word_dist(t) for t in range(lda.k)])

## Train PLDA Model
for epoch in tqdm(range(500), desc="PLDA Training"):
    plda.train(1)
    likelihood[epoch,1] = plda.ll_per_word
    theta_plda[epoch] = np.vstack(plda.infer(docs_plda, iter=100)[0])
    phi_plda[epoch] = np.vstack([plda.get_topic_word_dist(t) for t in range(plda.k)])

## Plot Likelihood
plt.figure(figsize=(10,5.8))
plt.plot(likelihood[:,0], label="LDA")
plt.plot(likelihood[:,1], label="PLDA")
plt.xlabel("Training Epoch", fontweight="bold")
plt.ylabel("Log Likelihood Per Word", fontweight="bold")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

## Plot Traces for Phi
for m, (mphi,mdl) in enumerate(zip([phi_lda, phi_plda],["LDA","PLDA"])):
    fig, ax = plt.subplots(mphi.shape[1], 1, figsize=(10,5.8), sharex=True)
    for k in range(mphi.shape[1]):
        ax[k].plot(mphi[:,k,:])
        ax[k].set_ylabel("Parameter Value", fontweight="bold")
    ax[k].set_xlabel("Training Epoch", fontweight="bold")
    ax[0].set_title(f"{mdl} $\\phi$ Trace", fontweight="bold")
    fig.tight_layout()
    plt.show()

## Plot Sample Traces for Theta
fig, ax = plt.subplots(5, 2, sharex=False, figsize=(10,5.8))
for d, doc in enumerate(sorted(np.random.choice(N, 5, replace=False))):
    ax[d,0].plot(theta_lda[:,doc,:])
    ax[d,1].plot(theta_plda[:,doc,:])
    for i in range(2):
        ax[d,i].spines["right"].set_visible(False)
        ax[d,i].spines["top"].set_visible(False)
        ax[d,i].set_title(f"Document {doc} $\\theta$", loc="left", fontstyle="italic")
for m, mdl in enumerate(["LDA","PLDA"]):
    ax[-1, m].set_xlabel(f"{mdl} Training Epoch", fontweight="bold")
fig.tight_layout()
plt.show()

## Get Final Representations
X_latent_lda = np.vstack(lda.infer(docs_lda, iter=1000, together=True)[0])
X_latent_plda = np.vstack(plda.infer(docs_plda, iter=1000, together=True)[0])

## Isolate Latent Variables and Normalize
X_latent_plda = X_latent_plda[:,-plda.latent_topics:]

## Fit Model
source_train_ind = sorted(set(train_ind) & set(np.where(D==0)[0]))
lr_lda = LogisticRegression(); lr_lda.fit(X_latent_lda[source_train_ind], y[source_train_ind])
lr_plda = LogisticRegression(); lr_plda.fit(X_latent_plda[source_train_ind], y[source_train_ind])

## Make Test Predictions
y_test_lda = lr_lda.predict_proba(X_latent_lda)[:,1]
y_test_plda = lr_plda.predict_proba(X_latent_plda)[:,1]

## Score Model
print("~~~~~~ Test Set Performance ~~~~~~")
for d, domain in enumerate(["Source","Target","Overall"]):
    if d == 2:
        domain_test_ind = test_ind
    else:
        domain_test_ind = sorted(set(test_ind) & set(np.where(D==d)[0]))
    lda_f1 = metrics.f1_score(y[domain_test_ind], y_test_lda[domain_test_ind]>0.5)
    plda_f1 = metrics.f1_score(y[domain_test_ind], y_test_plda[domain_test_ind]>0.5)
    lda_auc = metrics.roc_auc_score(y[domain_test_ind], y_test_lda[domain_test_ind])
    plda_auc = metrics.roc_auc_score(y[domain_test_ind], y_test_plda[domain_test_ind])
    print(domain,"Domain")
    print("LDA:  AUC={:.4f}, F1={:.4f}".format(lda_auc, lda_f1))
    print("PLDA: AUC={:.4f}, F1={:.4f}".format(plda_auc, plda_f1))
