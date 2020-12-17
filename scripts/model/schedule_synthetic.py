
#########################
### Configuration
#########################

## Grid Parameters
USERNAME = "kharrigian"
MEMORY = 16

## Experiment Information
EXPERIMENT_DIR = "./data/results/synthetic/"
EXPERIMENT_NAME = "DGP-CovariateShift"
EXPERIMENT_TRIALS = 1
RANDOM_STATE = 42
DRY_RUN = False

## Choose Base Configuration (Reference for Fixed Parameters)
BASE_CONFIG = "./scripts/model/train_synthetic.json"

## Specify Simple Parameter Search Space
N = [1000] 
SIGMA_0 = [1]
P_DOMAIN = [0.5]
GAMMA = [20]
V = [100]
BETA = [None]

## Specifiy Theta Parameter Search Space
USE_RATIOS = True
SOURCE_THETA_RATIOS = [1, 2, 5, 10, 50, 100]
TARGET_THETA_RATIOS = [1, 2, 5, 10, 50, 100]
SOURCE_THETA_RATIO_MAX = 5 ## General Latent to Source Latent
TARGET_THETA_RATIO_MAX = 5 ## General Latent to Target Latent
THETA_FLOOR = 1e-5

## Specify Coeficient Parameter Search Space
K_COEF = 10

## Topic Modeling Search Space
N_SAMPLE = [100]

######################
### Imports
######################

## Standard Library
import os
import sys
import json
import subprocess
from uuid import uuid4
from copy import deepcopy

## External Library
import numpy as np
from tqdm import tqdm
from mhlib.util.helpers import flatten
from mhlib.util.logging import initialize_logger
from sklearn.datasets import make_classification
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression

######################
### Globals
######################

## Logging
LOGGER = initialize_logger()

######################
### Functions
######################

def get_coefficient_samples(K=1000,
                            random_state=42):
    """

    """
    if random_state is not None:
        np.random.seed(random_state)
    ## Initialize Storage
    C = np.zeros((K,3))
    ## Generate Samples
    for k in tqdm(range(K)):
        ## Generate Dataset
        X, y = make_classification(n_samples=100,
                                   n_features=3,
                                   n_informative=3,
                                   n_redundant=0,
                                   n_repeated=0,
                                   n_clusters_per_class=2,
                                   weights=[0.5,0.5],
                                   flip_y=0.01,
                                   class_sep=0.5)
        ## Standardize
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        ## Learn Coefficients
        lr = LogisticRegression(penalty="none", fit_intercept=False)
        lr.fit(X, y)
        ## Cache
        C[k] = lr.coef_[0]
    return C

def get_theta(multiplier,
              domain=0,
              floor=1e-5,
              favor_general=True):
    """
    
    """
    t_general = 1
    if favor_general:
        t_specific = 1 / multiplier
    else:
        t_specific = multiplier
    if domain == 0:
        theta = [t_specific, floor, t_general]
    else:
        theta = [floor, t_specific, t_general]
    return theta

def create_parameter_grid():
    """

    """
    ## Standard Parameters
    pgrid = {
        "N":N,
        "sigma_0":SIGMA_0,
        "p_domain":P_DOMAIN,
        "gamma":GAMMA,
        "V":V,
        "beta":BETA,
        "n_sample":N_SAMPLE
    }
    ## Theta Parameter Search Space
    if not USE_RATIOS:
        source_thetas = [get_theta(i, domain=0, floor=THETA_FLOOR, favor_general=False) for i in range(1,SOURCE_THETA_RATIO_MAX+1)[::-1]] + \
                        [get_theta(i, domain=0, floor=THETA_FLOOR, favor_general=True) for i in range(2,SOURCE_THETA_RATIO_MAX+1)]
        target_thetas = [get_theta(i, domain=1, floor=THETA_FLOOR, favor_general=False) for i in range(1,TARGET_THETA_RATIO_MAX+1)[::-1]] + \
                        [get_theta(i, domain=1, floor=THETA_FLOOR, favor_general=True) for i in range(2,TARGET_THETA_RATIO_MAX+1)]
    else:
        source_thetas = [get_theta(i, domain=0, floor=THETA_FLOOR, favor_general=False) for i in SOURCE_THETA_RATIOS[::-1]] + \
                        [get_theta(i, domain=0, floor=THETA_FLOOR, favor_general=True) for i in SOURCE_THETA_RATIOS[1:]]
        target_thetas = [get_theta(i, domain=1, floor=THETA_FLOOR, favor_general=False) for i in TARGET_THETA_RATIOS[::-1]] + \
                        [get_theta(i, domain=1, floor=THETA_FLOOR, favor_general=True) for i in TARGET_THETA_RATIOS[1:]]

    thetas = []
    for st in source_thetas:
        for tt in target_thetas:
            thetas.append([st, tt])
    pgrid["theta"] = thetas
    ## Coefficient Parameter Search Space
    coefs = get_coefficient_samples(K=K_COEF, random_state=RANDOM_STATE)
    coefs = [[list(c),list(c)] for c in coefs]
    pgrid["coef"] = coefs
    ## Random States
    pgrid["random_state"] = list(range(1, EXPERIMENT_TRIALS+1))
    return pgrid

def create_configurations(pgrid):
    """

    """
    ## Load Base Config
    with open(BASE_CONFIG,"r") as the_file:
        base_config = json.load(the_file)
    ## Create Configurations
    configs = []
    for pg in tqdm(ParameterGrid(pgrid), desc="Creating Configurations"):
        ## Copy Base
        pg_config = deepcopy(base_config)
        ## Update Output Directory
        pg_id = str(uuid4())
        pg_config["output_dir"] = f"{EXPERIMENT_DIR}/{EXPERIMENT_NAME}/".replace("//","/")
        pg_config["run_id"] = pg_id
        ## Update Parameters
        pg_config.update(pg)
        ## Cache
        configs.append((pg_id, pg_config))
    return configs

def write_bash_script(temp_dir,
                      config_id,
                      config):
    """

    """
    ## Write Config
    config_filename = os.path.abspath(f"{temp_dir}/{EXPERIMENT_NAME}_{config_id}.json")
    with open(config_filename,"w") as the_file:
        json.dump(config, the_file)
    ## Write Script
    script = """
    #!/bin/bash
    #$ -cwd
    #$ -S /bin/bash
    #$ -m eas
    #$ -e /home/kharrigian/gridlogs/python/{}.err
    #$ -o /home/kharrigian/gridlogs/python/{}.out
    #$ -pe smp 8
    #$ -l 'gpu=0,mem_free={}g,ram_free={}g'

    ## Move to Home Directory (Place Where Virtual Environments Live)
    cd /home/kharrigian/
    ## Activate Conda Environment
    source .bashrc
    conda activate bayesian-stats-final
    ## Move To Run Directory
    cd /export/fs03/a08/kharrigian/topic-model-domain-adaptation/
    ## Run Script
    python ./scripts/model/train_synthetic.py \
    {} \
    """.format(f"{EXPERIMENT_NAME}_{config_id}",
               f"{EXPERIMENT_NAME}_{config_id}",
               MEMORY,
               MEMORY,
               config_filename,
               ).strip()
    bash_filename = config_filename.replace(".json",".sh")
    with open(bash_filename,"w") as the_file:
        the_file.write("\n".join([i.lstrip() for i in script.split("\n")]))
    return bash_filename

def main():
    """

    """
    ## Get Parameter Grid
    pgrid = create_parameter_grid()
    ## Get Configurations
    configs = create_configurations(pgrid)
    if len(configs) > 2000:
        raise Exception("Only allowed to submit 2000 jobs at a time. Refactor Configuration.")
    ## Exit
    if DRY_RUN:
        print("Dry Run Complete. Config results in {} jobs".format(len(configs)))
        exit()
    ## Create Temporary Directory
    temp_dir = "./temp_{}/".format(uuid4())
    _ = os.mkdir(temp_dir)
    ## Write Files
    bash_files = []
    for cid, c in tqdm(configs, desc="Writing Executables"):
        c_filename = write_bash_script(temp_dir=temp_dir,
                                       config_id=cid,
                                       config=c)
        bash_files.append(c_filename)
    ## Schedule Jobs
    for job_file in bash_files:
        qsub_call = f"qsub {job_file}"
        job_id = subprocess.check_output(qsub_call, shell=True)
        job_id = int(job_id.split()[2])
        LOGGER.info(f"Scheduled Job: {job_id}")
    ## Done
    LOGGER.info("Scheduled {} Jobs Total. Script Complete.".format(len(bash_files)))

######################
### Execute
######################

if __name__ == "__main__":
    _ = main()