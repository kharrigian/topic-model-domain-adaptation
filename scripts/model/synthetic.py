
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
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from mhlib.util.helpers import flatten

######################
### Load Newsgroups Data
######################

## Load Newsgroups Data
electrionics_categories = [
  'comp.graphics',
  'comp.os.ms-windows.misc',
  'comp.sys.ibm.pc.hardware',
  'comp.sys.mac.hardware',
  'comp.windows.x',
]
science_categories = [
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'sci.electronics',
]
X, y = fetch_20newsgroups(categories=electrionics_categories+science_categories,
                          return_X_y=True)


######################
### Generate Synthetic Data
######################

## Specify Parameters
N = 1000
gamma = 100
p_d = 0.5
topic_coef_mu = [1,0,0.0,-1,2,-2]
topic_coef_var = [1, .5, 1, .5, .5, 1]
theta = np.array([[0.3, 0.4, 0, 0, 0.2, 0.1],
                  [0, 0, 0.25, 0.45, 0.1, 0.2]])
phi = np.array([[10, 10, 1, 1, 1, 2, 2],
                [1, 1, 10, 10, 1, 2, 2],
                [1, 1, 2, 10, 1, 10, 1],
                [1, 10, 2, 2, 1, 10, 1],
                [2, 2, 2, 2, 2, 2, 1],
                [1, 1, 1, 1, 10, 2, 10]])

## Normalize and Smooth Generators
theta = (theta + 1e-5) / (theta + 1e-5).sum(axis=1,keepdims=True)
phi = (phi + 1e-5) / (phi + 1e-5).sum(axis=1,keepdims=True)

## Storage
X = np.zeros((N, phi.shape[1]))
y = np.zeros(N)
domain = np.zeros(N)

## Generate Dataset
coefs = stats.multivariate_normal(topic_coef_mu, topic_coef_var).rvs()
for n in range(N):
    ## Sample Domain
    d = int(np.random.rand() < p_d)
    domain[n] = d
    ## Sample Word Count
    dn = stats.poisson(gamma).rvs()
    ## Sample Topic Mixture (Conditioned on Domain)
    theta_n = stats.dirichlet(theta[d]).rvs()[0]
    ## Sample Y
    pyn = 1 / (1 + np.exp(-coefs.dot(theta_n)))
    y[n] = int(np.random.rand() < pyn)
    ## Sample Tokens
    x = np.zeros(phi.shape[1])
    for _ in range(dn):
        ## Sample Topic
        z = np.random.choice(theta_n.shape[0], p=theta_n)
        ## Sample Word (Conditioned on Topic)
        w = np.nonzero(stats.multinomial(n=1,p=phi[z]).rvs()[0])[0][0]
        ## Update Document Term Matrix
        X[n,w] += 1

######################
### Fit Topic Models
######################

## Helper Function
doc_to_str = lambda x: flatten([[str(i)]*int(j) for i, j in enumerate(x)])

## Split Data
train_ind = list(range(int(N*.8)))
test_ind = list(range(int(N*.8),N))

## Initialize Models
lda = tp.LDAModel(k=6,
                  seed=42)
plda = tp.PLDAModel(latent_topics=6,
                    topics_per_label=6,
                    seed=42)

## Add Training Data
for n in train_ind:
    doc_n = doc_to_str(X[n])
    d_n = domain[n]
    lda.add_doc(doc_n)
    plda.add_doc(doc_n, [str(d_n)])

## Train Models
lda.train(1000)
plda.train(1000)

## Get Representations
X_latent_lda = np.zeros((N, lda.k))
X_latent_plda = np.zeros((N, plda.k))
for n in range(N):
    doc_n = doc_to_str(X[n])
    X_latent_lda[n], _ = lda.infer(lda.make_doc(doc_n))
    X_latent_plda[n], _ = plda.infer(plda.make_doc(doc_n, [str(domain[n])]))

## Isolate Latent Variables
X_latent_plda = X_latent_plda[:,-plda.latent_topics:]

## Fit Model
lr_lda = LogisticRegression(); lr_lda.fit(X_latent_lda[train_ind], y[train_ind])
lr_plda = LogisticRegression(); lr_plda.fit(X_latent_plda[train_ind], y[train_ind])

## Make Test Predictions
y_test_lda = lr_lda.predict_proba(X_latent_lda[test_ind])[:,1]
y_test_plda = lr_plda.predict_proba(X_latent_plda[test_ind])[:,1]

## Score Model
lda_auc = metrics.roc_auc_score(y[test_ind], y_test_lda)
plda_auc = metrics.roc_auc_score(y[test_ind], y_test_plda)
print("LDA AUC: {:.4f}".format(lda_auc))
print("PLDA AUC: {:.4f}".format(plda_auc))