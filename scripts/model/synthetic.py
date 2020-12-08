
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
sigma0 = 3
p_domain = 0.5
gamma = 20
theta = np.array([[0.7, 0.2, 0.1],
                  [0.2, 0.7, 0.1]])
phi = np.array([[10, 5, 1, 1, 2, 2],
                [1, 1, 10, 5, 2, 3],
                [5, 15, 5, 1, 4, 7]])
coef = np.array([[1, -0.5, 1],
                 [-0.5, 1, 1]])

## Normalization of Parameters
theta = theta * sigma0
phi = phi / phi.sum(axis=1,keepdims=True)

## Storage
X_latent = np.zeros((N,coef.shape[1]))
X = np.zeros((N, phi.shape[1]))
D = np.zeros(N)

## Sample 
for n in range(N):
    ## Sample Domain
    d = int(np.random.rand() < p_domain)
    D[n] = d
    ## Sample Document Topic Mixture (Conditioned on Domain)
    X_latent[n] = stats.dirichlet(theta[d]).rvs()
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
    
## Standardize X
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
fig, ax = plt.subplots(3, 2, sharex=True)
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

## Initialize Models
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

## Train Models
likelihood = np.zeros((500, 2)) * np.nan
for epoch in range(100):
    lda.train(1); likelihood[epoch,0] = lda.ll_per_word
for epoch in range(500):
    plda.train(1); likelihood[epoch,1] = plda.ll_per_word

plt.plot(likelihood[:,0])
plt.plot(likelihood[:,1])
plt.show()

## Get Representations
X_latent_lda = np.zeros((N, lda.k))
X_latent_plda = np.zeros((N, plda.k))
for n in range(N):
    doc_n = doc_to_str(X[n])
    X_latent_lda[n], _ = lda.infer(lda.make_doc(doc_n),iter=1000)
    X_latent_plda[n], _ = plda.infer(plda.make_doc(doc_n, [str(D[n])]),iter=1000)

## Isolate Latent Variables and Normalize
X_latent_plda = X_latent_plda[:,-plda.latent_topics:]

## Fit Model
lr_lda = LogisticRegression(); lr_lda.fit(X_latent_lda[train_ind], y[train_ind])
lr_plda = LogisticRegression(); lr_plda.fit(X_latent_plda[train_ind], y[train_ind])

## Make Test Predictions
y_test_lda = lr_lda.predict_proba(X_latent_lda[test_ind])[:,1]
y_test_plda = lr_plda.predict_proba(X_latent_plda[test_ind])[:,1]

## Score Model
lda_f1 = metrics.f1_score(y[test_ind], y_test_lda>0.5)
plda_f1 = metrics.f1_score(y[test_ind], y_test_plda>0.5)
lda_auc = metrics.roc_auc_score(y[test_ind], y_test_lda)
plda_auc = metrics.roc_auc_score(y[test_ind], y_test_plda)
print("LDA: AUC={:.4f}, F1={:.4f}".format(lda_auc, lda_f1))
print("PLDA: AUC={:.4f}, F1={:.4f}".format(plda_auc, plda_f1))
