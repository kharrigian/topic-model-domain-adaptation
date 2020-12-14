
#########################
### Configuration
#########################

## Grid Parameters
USERNAME = "kharrigian"
MEMORY = 42

## Experiment Information
EXPERIMENT_DIR = "./data/results/depression/prior/wolohan_clpsych/"
EXPERIMENT_NAME = "PLDA"

## Choose Base Configuration (Reference for Fixed Parameters)
BASE_CONFIG = "./scripts/model/train.json"

## Specifiy Parameter Search Space
PARAMETER_GRID = {
    "source":["wolohan"],
    "target":["clpsych"],
    "use_plda":[True],
    "k_latent":[50],
    "k_per_label":[20],
    "alpha":[0.001, 0.01, 0.1, 1, 5, 10, 50, 100],
    "beta":[0.001, 0.01, 0.1, 1, 5, 10, 50, 100],
}


## Training Script Parameters
PLOT_DOCUMENT_TOPIC = False
PLOT_TOPIC_WORD = False

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
from mhlib.util.helpers import flatten
from mhlib.util.logging import initialize_logger
from sklearn.model_selection import ParameterGrid

######################
### Globals
######################

## Logging
LOGGER = initialize_logger()

######################
### Functions
######################

def create_configurations():
    """

    """
    ## Load Base Config
    with open(BASE_CONFIG,"r") as the_file:
        base_config = json.load(the_file)
    ## Create Configurations
    configs = []
    for pg in ParameterGrid(PARAMETER_GRID):
        ## Copy Base
        pg_config = deepcopy(base_config)
        ## Update Output Directory
        pg_id = "_".join(f"{x}-{y}" for x, y in pg.items())
        pg_id = pg_id.replace(":","-").\
                replace("{","").\
                replace("}","").\
                replace("'","").\
                replace(" ","").\
                replace(",","")
        pg_outdir = f"{EXPERIMENT_DIR}/{EXPERIMENT_NAME}/{pg_id}/".replace("//","/")
        pg_config["output_dir"] = pg_outdir
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
    config_filename = os.path.abspath(f"{temp_dir}/{config_id}.json")
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
    python ./scripts/model/train.py \
    {} \
    {} \
    {} \
    """.format(f"{EXPERIMENT_NAME}_{config_id}",
               f"{EXPERIMENT_NAME}_{config_id}",
               MEMORY,
               MEMORY,
               config_filename,
               {True:"--plot_document_topic",False:""}.get(PLOT_DOCUMENT_TOPIC),
               {True:"--plot_topic_word",False:""}.get(PLOT_TOPIC_WORD)
               ).strip()
    bash_filename = config_filename.replace(".json",".sh")
    with open(bash_filename,"w") as the_file:
        the_file.write("\n".join([i.lstrip() for i in script.split("\n")]))
    return bash_filename

def main():
    """

    """
    ## Get Configurations
    configs = create_configurations()
    ## Create Temporary Directory
    temp_dir = "./temp_{}/".format(uuid4())
    _ = os.mkdir(temp_dir)
    ## Write Files
    bash_files = []
    for cid, c in configs:
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