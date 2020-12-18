# Robust User Representations for Cross Domain Depression Classification using PLDA

This repository contains code for evaluating the use of Partially-labeled LDA for performing domain adaptation in the context of mental health status inference.

## Requirements

All experiments were executed using Python 3.8 within a Conda environment. To install dependencies, you can run `pip install -r requirements.txt`.

## Data

Due to the sensitive nature of mental health, we are not able to directly provide access to any of the datasets explored in the study. For more information about each dataset (CLPsych, Topic-Restricted Text), please see the main README in https://github.com/kharrigian/emnlp-2020-mental-health-generalization.

For reference, each preprocessed dataset results in the following data objects cached in `data/raw/depression/DATASET_NAME/`

* `config.json` - Parameters passed to `scripts/acquire/build_depression_data.py`, cached as JSON dictionary.
* `data.npz` - Sparse CSR Document-Term Matrix (N x V) dimensionality. Represents number of times each vocab term v was used by user n.
* `filenames.txt` - Newline delimited file of strings representing the name of the file storing raw data for each user. Dimensionality (N,).
* `splits.txt` - Newline delimited file of strings "train", "dev", or "test" denoting which split each user is part of. Dimensionality (N,)
* `targets.txt` - Newline delimited file of integers 1 ("depressed") or 0 ("control") denoting which group each user is in. Dimensionality (N, )
* `users.txt` - Newline delimited file of strings denoting the ID of each user in the document-term matrix. Dimensionality (N,)
* `vocab.txt` - Newline delimited file of strings denoting the vocabulary term represented by each column of the document term matrix. Dimensionality (V, )

All data objects for a given dataset can be loaded using the following function:

```
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
```

## Preprocessing and Modeling

1. We tokenize and cache local copies of the `CLPsych 2015 Shared Task` and `Topic-Restricted Text` datasets using `scripts/acquire/build_depression_data.py`

2. Assuming datasets have been processed, we can compare distributions of each dataset using `scripts/model/compare_distributions.py`

3. For learning topic-based representations and training mental health classifiers, we can use the following call:

```
python scripts/model/train.py scripts/model/train.json --plot_document_topic --plot_topic_word
```

Note that `scripts/model/train.json` configures the dataset, topic model, and downstream classifier.

4. For running synthetic data experiments, you can run the following command:

```
python scripts/model/train_synthetic.py scripts/model/train_synthetic.json
```

5. **CLSP Grid Only.** To run multiple experiments simultaneously, we can take advantage of compute on the CLSP grid. For scheduling a grid search over multiple parameters (synthetic and real datasets), one can look toward `scripts/model/schedule.py` and `scripts/model/schedule_synthetic.py`. 