
#########################
### Configuration
#########################

## Grid Parameters
USERNAME = "kharrigian"
MEMORY = 42
DRY_RUN = False

## Experiment Information
EXPERIMENT_DIR = "./data/results/depression/test-schedule/"
EXPERIMENT_NAME = "PLDA"

## Choose Base Configuration (Reference for Fixed Parameters)
BASE_CONFIG = "./scripts/model/train.json"

## Specifiy Parameter Search Space
PARAMETER_GRID = {
    "source":["clpsych_deduped","multitask"],
    "target":["clpsych_deduped","multitask"],
    "use_plda":[True],
    "k_latent":[50],
    "k_per_label":[20],
    "alpha":[0.1],
    "beta":[0.1],
    "topic_model_data":{
        "source":[None],
        "target":[None]
    }
}


## Training Script Parameters
PLOT_DOCUMENT_TOPIC = False
PLOT_TOPIC_WORD = False
K_FOLDS = 5
EVAL_TEST = False

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
    ## Topic-Model Sizes
    if "topic_model_data" in PARAMETER_GRID:
        PARAMETER_GRID["topic_model_data_source"] = PARAMETER_GRID["topic_model_data"]["source"]
        PARAMETER_GRID["topic_model_data_target"] = PARAMETER_GRID["topic_model_data"]["target"]
        _ = PARAMETER_GRID.pop("topic_model_data",None)
    ## Create Configurations
    configs = []
    for pg in ParameterGrid(PARAMETER_GRID):
        ## Check Dataset
        if pg["source"] == pg["target"]:
            continue
        if set([pg["source"],pg["target"]]) == set(["rsdd","smhd"]):
            continue
        ## Copy Base
        pg_config = deepcopy(base_config)
        ## Topic Model Data
        if "topic_model_data_source" in pg and "topic_model_data_target" in pg:
            pg["topic_model_data"] = {"source":pg.get("topic_model_data_source"),"target":pg.get("topic_model_data_target")}
            s = pg.pop("topic_model_data_source",None)
            t = pg.pop("topic_model_data_target",None)
            if s == 0 and t == 0:
                continue
            if (s == 0 or t == 0) and pg.get("use_plda", None):
                continue
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
                      config,
                      fold=None):
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
    conda activate plda-da
    ## Move To Run Directory
    cd /export/c01/kharrigian/topic-model-domain-adaptation/
    ## Run Script
    python ./scripts/model/train.py \
    {} \
    {} \
    {} \
    {} \
    {}
    """.format(f"{EXPERIMENT_NAME}_{config_id}",
               f"{EXPERIMENT_NAME}_{config_id}",
               MEMORY,
               MEMORY,
               config_filename,
               {True:"--plot_document_topic",False:""}.get(PLOT_DOCUMENT_TOPIC),
               {True:"--plot_topic_word",False:""}.get(PLOT_TOPIC_WORD),
               "--fold {}".format(fold) if fold is not None else "",
               {True:"--evaluate_test",False:""}.get(EVAL_TEST)
               ).strip()
    if fold is None:
        bash_filename = config_filename.replace(".json",".sh")
    else:
        bash_filename = config_filename.replace(".json",f"_fold-{fold}.sh")
    with open(bash_filename,"w") as the_file:
        the_file.write("\n".join([i.lstrip() for i in script.split("\n")]))
    return bash_filename

def main():
    """

    """
    ## Get Configurations
    configs = create_configurations()
    ## Cross Validation Folds
    if K_FOLDS is not None:
        folds = list(range(1, K_FOLDS+1))
    else:
        folds = [None]
    ## Create Temporary Directory
    temp_dir = "./temp_{}/".format(uuid4())
    _ = os.mkdir(temp_dir)
    ## Write Files
    bash_files = []
    for cid, c in configs:
        for f in folds:
            c_filename = write_bash_script(temp_dir=temp_dir,
                                           config_id=cid,
                                           config=c,
                                           fold=f)
            bash_files.append(c_filename)
    ## Dry Run Check
    if DRY_RUN:
        LOGGER.info("Identified {} Jobs".format(len(bash_files)))
        _ = os.system("rm -rf {}".format(temp_dir))
        LOGGER.info("Dry run complete. Exiting.")
        exit()
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