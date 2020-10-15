# We will automate all the logistic regression baseline experiments for different event types and their subtasks
# For each Event type we will first run the data_preprocessing and
# then run the logistic regression classifier for each subtask that has few (non-zero) positive examples
# We will save all the different classifier models, configs and results in separate directories
# Finally when all the codes have finished we will aggregate all the results and save the final metrics in csv file

from model.utils import make_dir_if_not_exists, load_from_pickle, load_from_json, MIN_POS_SAMPLES_THRESHOLD
import os
import json
import time
import csv
import subprocess

from produce_ouputs import main
import logging


def run_autotest(taskname,POSSIBLE_BATCH_SIZE=8,
                 use_gpu_if_possible=False):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    task_type_to_datapath_dict = {
        "tested_positive": ("./data/positive-add_text.jsonl", "./data/test_data/positive.pkl"),
        "tested_negative": ("./data/negative-add_text.jsonl", "./data/test_data/negative.pkl"),
        "can_not_test": ("./data/can_not_test-add_text.jsonl", "./data/test_data/can_not_test.pkl"),
        "death": ("./data/death-add_text.jsonl", "./data/test_data/death.pkl"),
        "cure": ("./data/cure_and_prevention-add_text.jsonl", "./data/test_data/cure_and_prevention.pkl"),
    }

    # REDO_DATA_FLAG = True
    REDO_DATA_FLAG = False
    REDO_FLAG = True
    RETRAIN_FLAG = True
    # REDO_FLAG = False

    # We will save all the tasks and subtask's results and model configs in this dictionary
    all_task_results_and_model_configs = dict()
    # We will save the list of question_tags AKA subtasks for each event AKA task in this dict
    all_task_question_tags = dict()
    data_in_file = task_type_to_datapath_dict[taskname][0]
    processed_out_file = task_type_to_datapath_dict[taskname][1]
    # for taskname, (data_in_file, processed_out_file) in task_type_to_datapath_dict.items():
    if not os.path.exists(processed_out_file) or REDO_DATA_FLAG:
        # data_preprocessing_cmd = f"python model/data_preprocessing.py -d {data_in_file} -s {processed_out_file}"
        # logging.info(data_preprocessing_cmd)
        # os.system(data_preprocessing_cmd)
        print('files not found')
        exit(0)
    else:
        logging.info(f"Preprocessed data for task {taskname} already exists at {processed_out_file}")

        # Read the data statistics
        # task_instances_dict, tag_statistics, question_keys_and_tags = load_from_pickle(processed_out_file)

        # We will store the list of subtasks for which we train the classifier
        tested_tasks = list()
        logging.info(f"Training Mutlitask BERT Entity Classifier model on {processed_out_file}")
        # output_dir = os.path.join("results", "multitask_bert_entity_classifier", taskname)
        # NOTE: After fixing the USER and URL tags
        # output_dir = os.path.join("results", "multitask_bert_entity_classifier_fixed", taskname)
        output_dir ='output_'+ taskname + '.jsonl'
        # make_dir_if_not_exists(output_dir)
        # results_file = os.path.join(output_dir, "results.json")
        # model_config_file = os.path.join(output_dir, "model_config.json")

        # Execute the Bert entity classifier train and test only if the results file doesn't exists
        # multitask_bert_cmd = f"python model/multitask_bert_entitity_classifier.py -d {processed_out_file} -t {taskname} -o {output_dir} -s saved_models/multitask_bert_entity_classifier/{taskname}_8_epoch_32_batch_multitask_bert_model"
        # After fixing the USER and URL tags
        inputs = {'data_file': processed_out_file, 'task': taskname,
                  'model_path': f'saved_models/multitask_bert_entity_classifier_fixed/{taskname}_8_epoch_32_batch_multitask_bert_model',
                  'output_dir': output_dir,
                   'batch_size': 32,
                  'POSSIBLE_BATCH_SIZE': POSSIBLE_BATCH_SIZE,
                  'use_gpu_if_possible': use_gpu_if_possible}

        multitask_bert_cmd = f"python model/multitask_bert_entitity_classifier.py -d {processed_out_file} -t {taskname} -o {output_dir} -s saved_models/multitask_bert_entity_classifier_fixed/{taskname}_8_epoch_32_batch_multitask_bert_model"
        # if RETRAIN_FLAG:
        #     multitask_bert_cmd += " -r"
        logging.info(f"Running: {multitask_bert_cmd}")
        main(**inputs)
        # try:
        #     retcode = subprocess.call(multitask_bert_cmd, shell=True)
        #     # os.system(multitask_bert_cmd)
        # except KeyboardInterrupt:
        #     exit()
