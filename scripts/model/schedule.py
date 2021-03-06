
#########################
### Configuration
#########################

## Grid Parameters
USERNAME = "kharrigian"
MEMORY = 72
DRY_RUN = False
LOG_DIR="/home/kharrigian/gridlogs/python/plda/train/optimized/"

## Experiment Information
EXPERIMENT_DIR = "./data/results/depression/optimized/"
EXPERIMENT_NAME = "PLDA"

## Choose Base Configuration (Reference for Fixed Parameters)
BASE_CONFIG = "./scripts/model/train.json"

## Option 1: Specify Parameter Search Space
# PARAMETER_GRID = {
#     "source":["clpsych_deduped","multitask","wolohan","smhd"],
#     "target":["clpsych_deduped","multitask","wolohan","smhd"],
#     "use_plda":[False],
#     "k_latent":[25,50,75,100,150,200],
#     "k_per_label":[25,50,75,100],
#     "alpha":[1e-2],
#     "beta":[1e-2],
#     "topic_model_data":{
#         "source":[None],
#         "target":[None]
#     }
# }

## Option 2: Specify Set of Experiments Using External File
# PARAMETER_GRID = "/export/c01/kharrigian/topic-model-domain-adaptation/scripts/model/optimized-lda.jl"
PARAMETER_GRID = "/export/c01/kharrigian/topic-model-domain-adaptation/scripts/model/optimized-plda.jl"

## Training Script Parameters
CACHE_PREDICTIONS = True
PLOT_DOCUMENT_TOPIC = False
PLOT_TOPIC_WORD = False
K_FOLDS = 5
EVAL_TEST = True

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

## Initialize Log Directiory
if not os.path.exists(LOG_DIR):
    _ = os.makedirs(LOG_DIR)

## Parameter Caching
CACHE_PARAMETERS = PLOT_DOCUMENT_TOPIC or PLOT_TOPIC_WORD

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
    if isinstance(PARAMETER_GRID, dict) and "topic_model_data" in PARAMETER_GRID:
        PARAMETER_GRID["topic_model_data_source"] = PARAMETER_GRID["topic_model_data"]["source"]
        PARAMETER_GRID["topic_model_data_target"] = PARAMETER_GRID["topic_model_data"]["target"]
        _ = PARAMETER_GRID.pop("topic_model_data",None)
    ## Get Grid
    if isinstance(PARAMETER_GRID,dict):
        pgrid = ParameterGrid(PARAMETER_GRID)
    elif isinstance(PARAMETER_GRID,str) and os.path.exists(PARAMETER_GRID):
        with open(PARAMETER_GRID,"r") as the_file:
            pgrid = json.load(the_file)
    else:
        raise ValueError("Parameter Grid not recognized")
    ## Create Configurations
    configs = []
    for pg in pgrid:
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
                replace("]","").\
                replace("[","").\
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
    #$ -e {}{}.err
    #$ -o {}{}.out
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
    {}
    """.format(LOG_DIR, f"{EXPERIMENT_NAME}_{config_id}",
               LOG_DIR, f"{EXPERIMENT_NAME}_{config_id}",
               MEMORY,
               MEMORY,
               config_filename,
               {True:"--plot_document_topic",False:""}.get(PLOT_DOCUMENT_TOPIC),
               {True:"--plot_topic_word",False:""}.get(PLOT_TOPIC_WORD),
               {True:"--evaluate_test",False:""}.get(EVAL_TEST)
               ).strip()
    bash_filename = config_filename.replace(".json",".sh")
    with open(bash_filename,"w") as the_file:
        the_file.write("\n".join([i.lstrip() for i in script.split("\n")]))
    return bash_filename

def write_array_bash_script(temp_dir,
                            config_id,
                            config,
                            k_folds):
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
    #$ -N {}
    #$ -t 1-{}
    #$ -e {}
    #$ -o {}
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
    --fold $SGE_TASK_ID \
    {} \
    {} \
    {} \
    {}
    """.format(f"{EXPERIMENT_NAME}_{config_id}",
               k_folds,
               LOG_DIR,
               LOG_DIR,
               MEMORY,
               MEMORY,
               config_filename,
               {True:"--plot_document_topic",False:""}.get(PLOT_DOCUMENT_TOPIC),
               {True:"--plot_topic_word",False:""}.get(PLOT_TOPIC_WORD),
               "--k_folds {}".format(k_folds),
               {True:"--evaluate_test",False:""}.get(EVAL_TEST),
               {True:"--cache_parameters",False:""}.get(CACHE_PARAMETERS),
               {True:"--cache_predictions",False:""}.get(CACHE_PREDICTIONS)
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
    ## Dry Run Check
    if DRY_RUN:
        LOGGER.info("Identified {} Jobs".format(len(configs)))
        LOGGER.info("Dry run complete. Exiting.")
        exit()
    ## Create Temporary Directory
    temp_dir = "./temp_{}/".format(uuid4())
    _ = os.mkdir(temp_dir)
    ## Write Files
    bash_files = []
    for cid, c in configs:
        if K_FOLDS is None:
            c_filename = write_bash_script(temp_dir=temp_dir,
                                           config_id=cid,
                                           config=c)
        else:
            c_filename = write_array_bash_script(temp_dir=temp_dir,
                                                 config_id=cid,
                                                 config=c,
                                                 k_folds=K_FOLDS)
        bash_files.append(c_filename)
    ## Schedule Jobs
    for job_file in bash_files:
        qsub_call = f"qsub {job_file}"
        job_id = subprocess.check_output(qsub_call, shell=True)
        job_id = job_id.split()[2]
        LOGGER.info(f"Scheduled Job: {job_id}")
    ## Done
    LOGGER.info("Scheduled {} Jobs Total. Script Complete.".format(len(bash_files)))

######################
### Execute
######################

if __name__ == "__main__":
    _ = main()