
"""
Create document-term matrices with associated
labels for a target dataset. Also defines 
splits for evaluation/training.
"""

## Where Standardized Data Formats Live
MH_DATA_DIR = "/export/fs03/a08/kharrigian/mental-health/data/processed/"
PLDA_DIR = "/export/c01/kharrigian/topic-model-domain-adaptation/"

#####################
### Imports
#####################

## Standard Library
import os
import sys
import json
import gzip
import argparse
from glob import glob
from functools import partial
from collections import Counter
from multiprocessing.dummy import Pool

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import vstack, save_npz
from sklearn.feature_extraction import DictVectorizer
from mhlib.util.helpers import flatten, chunks
from mhlib.preprocess.tokenizer import Tokenizer, get_ngrams
from mhlib.util.logging import initialize_logger

#####################
### Globals
#####################

## Location of Semi-Processed Data
MH_DATA_SUFFIXES = {
    "clpsych":"twitter/qntfy/",
    "clpsych_deduped":"twitter/qntfy/",
    "multitask":"twitter/qntfy/",
    "rsdd":"reddit/rsdd/",
    "smhd":"reddit/smhd/",
    "wolohan":"reddit/wolohan/*comments*"
}

## Date Ranges
MH_DATA_DATE_BOUNDARIES = {
    "clpsych":("2011-01-01","2013-12-01"),
    "clpsych_deduped":("2011-01-01","2013-12-01"),
    "multitask":("2013-01-01","2016-01-01"),
    "rsdd":("2010-01-01","2017-01-01"),
    "smhd":("2011-01-01","2018-01-01"),
    "wolohan":("2016-01-01","2020-01-01")
}

## Where to Output Depression Data
CACHE_DIR = f"{PLDA_DIR}/data/raw/depression/".replace("//","/")

## Logging
LOGGER = initialize_logger()

## Tokenizer
TOKENIZER = Tokenizer(stopwords=set(),
                      keep_case=False,
                      negate_handling=True,
                      negate_token=False,
                      upper_flag=False,
                      keep_punctuation=False,
                      keep_numbers=False,
                      expand_contractions=True,
                      keep_user_mentions=False,
                      keep_pronouns=True,
                      keep_url=False,
                      keep_hashtags=True,
                      keep_retweets=False,
                      emoji_handling=None,
                      strip_hashtag=False)

## Vectorizer
cvec = None

#####################
### Functions
#####################

def parse_arguments():
    """
    Parse command-line arguments.

    Args:
        None
    
    Returns:
        args (argparse Object): Command-line argument holder.
    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="")
    ## Required Arguments
    parser.add_argument("dataset",
                        type=str,
                        help="Colloquial name of the dataset")
    ## Optional Arguments
    parser.add_argument("--remove_retweets",
                        action="store_true",
                        default=False)
    parser.add_argument("--binarize_vocab",
                        action="store_true",
                        default=False)
    parser.add_argument("--pretokenize",
                        action="store_true",
                        default=False)
    parser.add_argument("--chunksize",
                        default=100,
                        type=int)
    parser.add_argument("--prune_rate",
                        default=1,
                        type=int)
    parser.add_argument("--min_prune_freq",
                        default=3,
                        type=int)
    parser.add_argument("--min_n",
                        default=1,
                        type=int)
    parser.add_argument("--max_n",
                        default=1,
                        type=int)
    parser.add_argument("--jobs",
                        default=4,
                        type=int,
                        help="Number of processes to use for data vectoriztion.")
    ## Parse Arguments
    args = parser.parse_args()
    return args

def load_json_file(filename):
    """

    """
    with gzip.open(filename,"r") as the_file:
        data = json.load(the_file)
    return data

def get_label_map(dataset):
    """

    """
    ## Identifiy Metadata Files
    metadata_pattern = "{}{}*.meta.tar.gz".format(MH_DATA_DIR, MH_DATA_SUFFIXES.get(dataset))
    metadata_files = sorted(glob(metadata_pattern))
    ## Load Metadata, Construct Relevant Label Mapping
    label_map = {}
    for mf in tqdm(metadata_files, desc="Loading Metadata", file=sys.stdout):
        mf_data = load_json_file(mf)
        if mf_data.get("depression",None) not in set(["depression","control"]):
            continue
        if dataset not in mf_data.get("datasets",set()):
            continue
        mf_raw = mf.replace(".meta","")
        if not os.path.exists(mf_raw):
            continue
        label_map[mf_raw] = mf_data
    ## Format Label Map
    label_map = pd.DataFrame(label_map).T
    label_columns = ["user_id_str",
                     "age",
                     "gender",
                     "split",
                     "depression"]
    label_map = label_map[[l for l in label_columns if l in label_map.columns]].copy()
    label_map = label_map.reset_index().rename(columns={"index":"source"})
    if "age" in label_map.columns:
        label_map["age"] = label_map["age"].astype(float)
    return label_map

def _assign_value_to_bin(value,
                         bins):
    """
    Assign a value to a bin index based on a set of bins

    Args:
        value (numeric): Value for assignment
        bins (list): Bin boundaries. [a,b) bins will be considered
    
    Returns:
        b (int): Index of bin boundaries where value lies.
    """
    if value < bins[0]:
        return -1
    b = 0
    for bstart, bend in zip(bins[:-1], bins[1:]):
        if value >= bstart and value < bend:
            return b
        b += 1
    return b

def _generate_stratified_sample(x,
                                y,
                                n,
                                bins=100):
    """
    Given a population represented by x, select a subset of a population
    represented by y that looks like population x

    Args:
        x (array): Representative values (numeric) for population x
        y (array): Representative values (numeric) for population y
        n (int): How many samples to select from population y
        bins (int): How fine the binning should be for quantifying the population of x
    
    Returns:
        y_sample (list): List of integer indices from sample y that will give
                         a representative sample similar to that parameterized by x
    """
    ## Bin The Input Sample To Match (x)
    x_counts, x_bins = np.histogram(x, bins=bins)
    ## Compute Selection Probability, With 0 probability assigned to below and above extrema
    p_x = np.array([0] + list(x_counts / x_counts.sum()) + [0])
    ## Bin the Y Values, Adding 1 to Account for Extrema Probabilities
    y_binned = np.array(list(map(lambda v: _assign_value_to_bin(v, x_bins), y))) + 1
    y_p_select = p_x[y_binned]
    ## Normalize Sample Probabilities
    y_p_select = y_p_select / y_p_select.sum()
    ## Sample y
    m = len(y)
    y_sample = np.random.choice(m, size=n, replace=False, p=y_p_select)
    ## Sort Sample
    y_sample = sorted(y_sample)
    return y_sample

def _sample_by_age_and_gender(label_map,
                              genders=["M","F"],
                              random_state=42):
    """
    Identify a matched sample of users in a target disorder set based 
    on data set split, age, and gender

    Args:
        metadata_df (pandas DataFrame): Label data
        genders (list): List of genders to split on.
        random_state (int): Random seed for sampling
    
    Returns:
        matched_data (pandas DataFrame): Balanced sample of data
    """    
    ## Set Random Sample Seed
    np.random.seed(random_state) 
    ## Train/Dev/Test Split Preservation
    metadata_df = label_map.copy()
    if "split" in metadata_df.columns and len(metadata_df["split"].unique()) > 1:
        splits = metadata_df["split"].unique()
    else:
        splits = ["all"]
        metadata_df["split"] = "all"
    ## Separate Groups
    target_disorder_group = metadata_df.loc[metadata_df["depression"] == "depression"]
    control_group = metadata_df.loc[metadata_df["depression"] == "control"]
    ## Sample Population
    matched_data = []
    for split in splits:
        ## Cycle Through Genders
        for gender in genders:
            ## Isolate Pool of Users
            gender_target_disorder_pool = target_disorder_group.loc[(target_disorder_group["gender"]==gender)&
                                                                    (target_disorder_group["split"]==split)]
            gender_control_pool = control_group.loc[(control_group["gender"]==gender)&
                                                    (control_group["split"]==split)]
            ## Sample Control Population Based on Age
            gender_control_sample = _generate_stratified_sample(gender_target_disorder_pool["age"].values,
                                                                gender_control_pool["age"].values,
                                                                n = len(gender_target_disorder_pool),
                                                                bins = 100)
            matched_data.append(gender_control_pool.iloc[gender_control_sample])
            matched_data.append(gender_target_disorder_pool)
    matched_data = pd.concat(matched_data).reset_index(drop=True).copy()
    return matched_data

def get_dataset_splits(label_map,
                       dev_size=.2,
                       test_size=.2,
                       random_state=42):
    """
    Three Cases:
    - Dataset does not have predefined splits -> Generate from scratch
    - Dataset has predefined train/dev/test splits -> No change
    - Dataset has predefined train/test splits -> Split train

    Args:
        label_map (pandas DataFrame)
        dev_size (float (0,1))
        test_size (float (0,1))
        random_state (int or None)
    
    Returns:
        user_id_map (dict)
    """
    ## Preprocessing: Population Balancing
    if "age" in label_map.columns and "gender" in label_map.columns:
        label_map = _sample_by_age_and_gender(label_map, random_state=random_state)
    ## Reset Random Seed
    if random_state is not None:
        np.random.seed(random_state)
    ## Case 1: No Splits
    if "split" not in label_map.columns or label_map["split"].unique()[0] == "all":
        test_ind = np.random.choice(label_map.index,
                                    size=int(label_map.shape[0] * test_size),
                                    replace=False)
        dev_ind = np.random.choice(label_map.loc[~label_map.index.isin(test_ind)].index,
                                   size=int((label_map.shape[0]-len(test_ind)) * dev_size),
                                   replace=False)
        train_ind = label_map.loc[(~label_map.index.isin(test_ind))&
                                  (~label_map.index.isin(dev_ind))].index.values
    ## Case 2/3: Splits Exist
    else:
        ## Case 2: Train/Dev/Test
        if all(i in label_map["split"].values for i in ["train","dev","test"]):
            test_ind = label_map.loc[label_map["split"]=="test"].index.values
            dev_ind = label_map.loc[label_map["split"]=="dev"].index.values
            train_ind = label_map.loc[label_map["split"]=="train"].index.values
        ## Case 3: Train/Test
        else:
            test_ind = label_map.loc[label_map["split"]=="test"].index.values
            dev_ind = np.random.choice(label_map.loc[~label_map.index.isin(test_ind)].index,
                                       size=int((label_map.shape[0]-len(test_ind)) * dev_size),
                                       replace=False)
            train_ind = label_map.loc[(~label_map.index.isin(test_ind))&
                                      (~label_map.index.isin(dev_ind))].index.values
    ## Refilter Label Map
    all_inds = sorted(set(train_ind) | set(dev_ind) | set(test_ind))
    label_map = label_map.loc[all_inds].copy()
    label_map["split"] = np.nan
    label_map.loc[train_ind, "split"] = "train"
    label_map.loc[dev_ind, "split"] = "dev"
    label_map.loc[test_ind, "split"] = "test"
    ## Get User ID Map
    user_id_map = {
        "train":label_map.loc[sorted(train_ind)]["user_id_str"].tolist(),
        "dev":label_map.loc[sorted(dev_ind)]["user_id_str"].tolist(),
        "test":label_map.loc[sorted(test_ind)]["user_id_str"].tolist()
    }
    ## Update Label Map Index
    label_map = label_map.reset_index(drop=True)
    return label_map, user_id_map

def load_and_tokenize(filename,
                      min_n=1,
                      max_n=1,
                      min_date=None,
                      max_date=None,
                      remove_retweets=False,
                      cache_dir=None,
                      pretokenized=False):
    """

    """
    ## Load Data
    data = load_json_file(filename)
    ## Apply Tokenization
    if not pretokenized:
        ## Retweet Filtering
        if remove_retweets:
            data = [i for i in data if "RT @" not in i.get("text")]
        ## Date Filtering
        if min_date is None:
            min_date = min(i.get("created_utc") for i in data) if len(data) > 0 else -1
        if max_date is None:
            max_date = max(i.get("created_utc") for i in data) if len(data) > 0 else np.inf
        data = [i for i in data if i.get("created_utc")>=min_date and i.get("created_utc")<=max_date]
        ## Tokenize Text
        tokens = list(map(TOKENIZER.tokenize, [i["text"] for i in data]))
        ## Get Ngrams
        ngrams = list(map(lambda x: get_ngrams(x, min_n, max_n), tokens))
    else:
        ## Isolate Ngrams from Pretokenized Data
        ngrams = [list(map(tuple, i["text"])) for i in data]
    ## Return
    if cache_dir is None:
        return ngrams
    ## Cache
    ngrams = [{"text":n} for n in ngrams]
    cache_file = f"{cache_dir}/{os.path.basename(filename)}".replace("//","/")
    with gzip.open(cache_file,"wt") as the_file:
        json.dump(ngrams, the_file)
    return (filename, cache_file)

def tokenize_and_count(filename,
                       min_n=1,
                       max_n=1,
                       min_date=None,
                       max_date=None,
                       remove_retweets=True,
                       pretokenized=False):
    """
    Args:
        filename (str):
        min_n (int)
        max_n (int)
        min_date (None or int)
        max_date (None or int)
        remove_retweets (bool)
        pretokenized (bool)
    
    Returns:
        token_counts (Counter): Count of n-grams
    """
    ## Get Ngrams
    ngrams = load_and_tokenize(filename,
                               min_n=min_n,
                               max_n=max_n,
                               min_date=min_date,
                               max_date=max_date,
                               remove_retweets=remove_retweets,
                               pretokenized=pretokenized,
                               cache_dir=None)
    ## Count
    token_counts = Counter(flatten(ngrams))
    return token_counts

def apply_tokenizer(filenames,
                    cache_dir,
                    min_n=1,
                    max_n=1,
                    min_date=None,
                    max_date=None,
                    remove_retweets=False,
                    jobs=4):
    """

    """
    ## Tokenizer
    helper = partial(load_and_tokenize,
                     min_n=min_n,
                     max_n=max_n,
                     min_date=min_date,
                     max_date=max_date,
                     remove_retweets=remove_retweets,
                     cache_dir=cache_dir,
                     pretokenized=False)
    ## Initialize Pool
    mp = Pool(jobs)
    filenames = list(tqdm(mp.imap_unordered(helper, filenames),
                          desc="Tokenizer",
                          total=len(filenames),
                          file=sys.stdout))
    _ = mp.close()
    ## Filename Map
    filenames = dict((y,x) for x, y in filenames)
    ## Return Filenames
    return filenames

def learn_vocabulary(filenames,
                     chunksize=100,
                     prune_rate=1,
                     min_prune_freq=1,
                     min_date=None,
                     max_date=None,
                     min_n=1,
                     max_n=1,
                     remove_retweets=True,
                     binarize=False,
                     jobs=4,
                     pretokenized=False):
    """
    Args:
        filenames (list of str): Raw data filenames
        chunksize (int): How many files to process in parallel before aggregating
        prune_rate (int): How many chunks to process before pruning vocabulary
        min_prune_freq (int): N-grams below this threshold at each pruning are removed
    """
    ## Storage
    vocab = Counter()
    agg_chunk_counts = Counter()
    ## Initialize Multiprocessor
    mp = Pool(jobs)
    ## Initialize Tokenize/Count Function
    counter = partial(tokenize_and_count,
                      min_date=min_date,
                      max_date=max_date,
                      min_n=min_n,
                      max_n=max_n,
                      remove_retweets=remove_retweets,
                      pretokenized=pretokenized)
    ## Process Data
    filechunks = list(chunks(filenames, chunksize))
    chunks_processed = 0
    for chunk in tqdm(filechunks, desc="Learning Vocabulary", file=sys.stdout, position=0):
        ## Increment Chunk Count
        chunks_processed += 1
        ## Apply Counter
        chunk_counts = list(tqdm(mp.imap_unordered(counter, chunk), total=len(chunk), desc="File", file=sys.stdout, position=1, leave=False))
        ## Aggregate Counts
        for cc in chunk_counts:
            if binarize:
                cc = Counter({i:1 for i in cc.keys()})
            agg_chunk_counts += cc
        ## Update Cache
        if chunks_processed == prune_rate:
            ## Apply Pruning
            agg_chunk_counts = Counter({x:y for x, y in agg_chunk_counts.items() if y >= min_prune_freq})
            ## Add to General Vocab
            vocab += agg_chunk_counts
            ## Reset Chunk Counter
            agg_chunk_counts = Counter()
            chunks_processed = 0
    ## Apply Final Update
    if len(agg_chunk_counts) > 0:
        ## Prune Filter
        agg_chunk_counts = Counter({x:y for x, y in agg_chunk_counts.items() if y >= min_prune_freq})
        ## Add to General Vocab
        vocab += agg_chunk_counts
    ## Close Pool
    _ = mp.close()
    ## Initialize Vectorizer using Learned Vocabulary
    global cvec
    cvec = initialize_vectorizer(vocab)
    return vocab

def initialize_vectorizer(vocabulary):
    """
    Initialize a vectorizer that transforms a counter dictionary
    into a sparse vector of counts (with a uniform feature index)
    """
    ## Isolate Terms, Sort Alphanumerically
    terms = sorted(list(vocabulary.keys()))
    ngram_to_idx = dict((t, i) for i, t in enumerate(terms))
    ## Create Dict Vectorizer
    _count2vec = DictVectorizer(separator=":",dtype=int)
    _count2vec.vocabulary_ = ngram_to_idx.copy()
    rev_dict = dict((y, x) for x, y in ngram_to_idx.items())
    _count2vec.feature_names_ = [rev_dict[i] for i in range(len(rev_dict))]
    return _count2vec

def _vectorize_file(filename,
                    min_date=None,
                    max_date=None,
                    min_n=1,
                    max_n=1,
                    remove_retweets=False,
                    pretokenized=False):
    """

    """
    file_counts = tokenize_and_count(filename,
                                     min_date=min_date,
                                     max_date=max_date,
                                     min_n=min_n,
                                     max_n=max_n,
                                     remove_retweets=remove_retweets,
                                     pretokenized=pretokenized)
    x = cvec.transform(file_counts)
    return filename, x

def vectorize_files(filenames,
                    min_date=None,
                    max_date=None,
                    min_n=1,
                    max_n=1,
                    remove_retweets=False,
                    jobs=4,
                    pretokenized=False):
    """

    """
    ## Initialize Helper
    vectorizer = partial(_vectorize_file,
                         min_date=min_date,
                         max_date=max_date,
                         min_n=min_n,
                         max_n=max_n,
                         remove_retweets=remove_retweets,
                         pretokenized=pretokenized)
    ## Vectorize using Multiprocessing
    mp = Pool(jobs)
    results = list(tqdm(mp.imap_unordered(vectorizer, filenames), total=len(filenames), desc="Vectorizing Files", file=sys.stdout))
    _ = mp.close()
    ## Parse Results
    filenames = [r[0] for r in results]
    X = vstack(r[1] for r in results)
    return filenames, X

def main():
    """

    """
    ## Parse Command Line
    args = parse_arguments()
    ## Establish Cache Directory
    dataset_cache_dir = f"{CACHE_DIR}{args.dataset}/"
    if not os.path.exists(dataset_cache_dir):
        _ = os.makedirs(dataset_cache_dir)
    ## Isolate Preprocessing Arguments
    preprocessing_kwargs = dict( 
                     min_date=int(pd.to_datetime(MH_DATA_DATE_BOUNDARIES.get(args.dataset)[0]).timestamp()),
                     max_date=int(pd.to_datetime(MH_DATA_DATE_BOUNDARIES.get(args.dataset)[1]).timestamp()),
                     min_n=args.min_n,
                     max_n=args.max_n,
                     remove_retweets=args.remove_retweets,
                     jobs=args.jobs)
    ## Get Metadata/Labels
    label_map = get_label_map(dataset=args.dataset)
    ## Get Data Splits
    label_map, data_splits = get_dataset_splits(label_map,
                                     dev_size=.2,
                                     test_size=.2,
                                     random_state=42)
    ## Pretokenize if Desired
    if args.pretokenize:
        ## Create Temporary Directory
        temp_token_dir = f"{dataset_cache_dir}temp_tokenized/"
        if not os.path.exists(temp_token_dir):
            _ = os.makedirs(temp_token_dir)
        ## Apply Tokenizer
        temp_filenames = apply_tokenizer(filenames=label_map["source"].tolist(),
                                         cache_dir=temp_token_dir,
                                         **preprocessing_kwargs)
    ## Learn Vocabulary
    vocabulary = learn_vocabulary(filenames=label_map["source"].tolist() if not args.pretokenize else list(temp_filenames.keys()),
                                  chunksize=args.chunksize,
                                  prune_rate=args.prune_rate,
                                  min_prune_freq=args.min_prune_freq,
                                  binarize=args.binarize_vocab,
                                  pretokenized=args.pretokenize,
                                  **preprocessing_kwargs)
    ## Extract Terms
    terms = ["_".join(i) for i in cvec.feature_names_]
    ## Vectorize The Data
    filenames, X = vectorize_files(filenames=label_map["source"].tolist() if not args.pretokenize else list(temp_filenames.keys()),
                                   pretokenized=args.pretokenize,
                                   **preprocessing_kwargs)
    ## Remamp Filenames
    if args.pretokenize:
        filenames = [temp_filenames[f] for f in filenames]
    ## Vectorize the Metadata
    y = (label_map.set_index("source").loc[filenames]["depression"]=="depression").astype(int).tolist()
    u = label_map.set_index("source").loc[filenames]["user_id_str"].tolist()
    s = label_map.set_index("source").loc[filenames]["split"].tolist()
    ## Cache Processed Data
    for datum, datum_name in zip([terms,filenames,y,u,s],
                                 ["vocab","filenames","targets","users","splits"]):
        datum_file = f"{dataset_cache_dir}{datum_name}.txt"
        with open(datum_file,"w") as the_file:
            for item in datum:
                the_file.write(f"{item}\n")
    save_npz(f"{dataset_cache_dir}data.npz", X)
    ## Cache Preprocessing Parameters
    preprocessing_kwargs.update(dict(
                     chunksize=args.chunksize,
                     prune_rate=args.prune_rate,
                     min_prune_freq=args.min_prune_freq,
                     binarize=args.binarize_vocab))
    with open(f"{dataset_cache_dir}config.json","w") as the_file:
        json.dump(preprocessing_kwargs, the_file)
    ## Remove Temporary Files
    if args.pretokenize:
        LOGGER.info("Removing Temporary Files")
        _ = os.system("rm -rf {}".format(temp_token_dir))

######################
### Execute
######################

if __name__ == "__main__":
    _ = main()