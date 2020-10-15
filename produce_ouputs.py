import torch
import logging
import os
from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig, AdamW, \
    get_linear_schedule_with_warmup
import random
import json
from model.utils import make_dir_if_not_exists, load_from_pickle, load_from_json, MIN_POS_SAMPLES_THRESHOLD
from model.utils import log_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, \
    get_multitask_instances_for_valid_tasks, split_multitask_instances_in_train_dev_test, log_data_statistics, \
    save_in_json, get_raw_scores, get_TP_FP_FN
from model.multitask_bert_entitity_classifier import MultiTaskBertForCovidEntityClassification, \
    split_data_based_on_subtasks, log_multitask_data_statistics, COVID19TaskDataset, TokenizeCollator, \
    make_predictions_on_dataset
from model import multitask_bert_entitity_classifier
from torch.utils.data import Dataset, DataLoader


def main(data_file, task, model_path, output_dir, batch_size=32, POSSIBLE_BATCH_SIZE=8,
         use_gpu_if_possible=False):
    multitask_bert_entitity_classifier.set_device(use_gpu_if_possible)
    global device
    device = multitask_bert_entitity_classifier.get_device()
    # Also add the stream handler so that it logs on STD out as well
    # Ref: https://stackoverflow.com/a/46098711/4535284
    # make_dir_if_not_exists(output_dir)
    # if retrain:
    #     logfile = os.path.join(output_dir, "train_output.log")
    # else:
    #     logfile = os.path.join(output_dir, "output.log")
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    #                     handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])

    task_instances_dict, tag_statistics, question_keys_and_tags = load_from_pickle(data_file)
    # print(task_instances_dict.keys())
    # for key in task_instances_dict.keys():
    #     task_instances_dict[key] = task_instances_dict[key][:20]
    #

    data, subtasks_list = get_multitask_instances_for_valid_tasks(task_instances_dict, tag_statistics, test_mode=True)
    # print(data)

    # Load the tokenizer and model from the save_directory
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = MultiTaskBertForCovidEntityClassification.from_pretrained(model_path)
    # print(model.state_dict().keys())
    # TODO save and load the subtask classifier weights separately
    # Load from individual state dicts
    print(model.subtasks)
    if task == 'tested_negative' and 'how_long' in model.subtasks:
        model.subtasks.remove('how_long')
    if task == 'death' and 'symptoms' in  model.subtasks:
        model.subtasks.remove('symptoms')
    for subtask in model.subtasks:
        model.classifiers[subtask].load_state_dict(
            torch.load(os.path.join(model_path, f"{subtask}_classifier.bin")))
    # print(model.config)
    # exit()
    model.to(device)
    # Explicitly move the classifiers to device
    for subtask, classifier in model.classifiers.items():
        classifier.to(device)
    entity_start_token_id = tokenizer.convert_tokens_to_ids(["<E>"])[0]
    entity_end_token_id = tokenizer.convert_tokens_to_ids(["</E>"])[0]

    logging.info(f"Task dataset for task: {task} loaded from {data_file}.")

    model_config = dict()
    results = dict()

    # Split the data into train, dev and test and shuffle the train segment
    test_data, dev_data, _ = split_multitask_instances_in_train_dev_test(data, TRAIN_RATIO=1.0, test_mode=True)
    # print(test_data)
    # random.shuffle(train_data)  # shuffle happens in-place
    # logging.info("Train Data:")
    # total_train_size, pos_subtasks_train_size, neg_subtasks_train_size = log_multitask_data_statistics(train_data,
    #                                                                                                    model.subtasks)
    # logging.info("Dev Data:")
    # total_dev_size, pos_subtasks_dev_size, neg_subtasks_dev_size = log_multitask_data_statistics(dev_data,
    #                                                                                              model.subtasks)
    # logging.info("Test Data:")
    # total_test_size, pos_subtasks_test_size, neg_subtasks_test_size = log_multitask_data_statistics(test_data,
    #                                                                                                 model.subtasks)
    # logging.info("\n")
    # model_config["train_data"] = {"size": total_train_size, "pos": pos_subtasks_train_size,
    #                               "neg": neg_subtasks_train_size}
    # model_config["dev_data"] = {"size": total_dev_size, "pos": pos_subtasks_dev_size, "neg": neg_subtasks_dev_size}
    # model_config["test_data"] = {"size": total_test_size, "pos": pos_subtasks_test_size, "neg": neg_subtasks_test_size}

    # Extract subtasks data for dev and test
    # dev_subtasks_data = split_data_based_on_subtasks(dev_data, model.subtasks)
    # test_subtasks_data = split_data_based_on_subtasks(test_data, model.subtasks)

    # Load the instances into pytorch dataset
    # train_data = train_data[:100]
    # dev_data = dev_data[:30]
    # test_data = test_data[:30]
    # train_dataset = COVID19TaskDataset(train_data)

    dev_dataset = COVID19TaskDataset(dev_data)
    test_dataset = COVID19TaskDataset(test_data)

    logging.info("Loaded the datasets into Pytorch datasets")

    tokenize_collator = TokenizeCollator(tokenizer, model.subtasks, entity_start_token_id, entity_end_token_id,
                                         test_mode=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=POSSIBLE_BATCH_SIZE, shuffle=True, num_workers=0,
    #                               collate_fn=tokenize_collator)
    dev_dataloader = DataLoader(dev_dataset, batch_size=POSSIBLE_BATCH_SIZE, shuffle=False, num_workers=0,
                                collate_fn=tokenize_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=POSSIBLE_BATCH_SIZE, shuffle=False, num_workers=0,
                                 collate_fn=tokenize_collator)
    logging.info("Created train and test dataloaders with batch aggregation")

    # Save the model name in the model_config file
    model_config["model"] = "MultiTaskBertForCovidEntityClassification"
    # model_config["epochs"] = n_epochs
    # print(test_dataloader.dataset.instances)
    # test_dataloader.dataset
    predicted_labels, prediction_scores, gold_labels, all_chunks, all_ids = make_predictions_on_dataset(test_dataloader,
                                                                                                        model, device,
                                                                                                        task,
                                                                                                        test_mode=True)

    length = 0
    for subtask in model.subtasks:
        length = len(prediction_scores[subtask])
        print(len(prediction_scores[subtask]), len(predicted_labels[subtask]), len(all_chunks[subtask]),
              len(all_ids[subtask]))
    # print(predicted_labels)
    output = {}
    for i in range(length):
        for subtask in model.subtasks:
            id = all_ids[subtask][i]
            chunk = all_chunks[subtask][i]
            if id not in output.keys():
                output[id] = {}
                output[id]['id'] = id
                output[id]['predicted_annotation'] = {}
            else:
                if prediction_scores[subtask][i] <= 0.5 and predicted_labels[subtask][i] == 1:
                    print('WHAT?', prediction_scores[subtask][i], predicted_labels[subtask][i])
                if subtask not in output[id]['predicted_annotation'].keys():
                    output[id]['predicted_annotation'][subtask] = [chunk, prediction_scores[subtask][i]]
                else:
                    already_score = output[id]['predicted_annotation'][subtask][1]
                    now_score = prediction_scores[subtask][i]
                    if now_score > already_score:
                        output[id]['predicted_annotation'][subtask] = [chunk, now_score]
    json_list = []
    for id in output.keys():
        anot = output[id]['predicted_annotation']
        json_rec = {}
        json_rec['id'] = id
        json_rec['predicted_annotation'] = {}

        for qtag in anot.keys():
            json_rec['predicted_annotation']['part2-' + qtag + '.Response'] = output[id]['predicted_annotation'][qtag][
                0]
        json_list.append(json_rec)
    # print(output_dir)
    # print(output)
    # print(json_list)
    # print(json_list[0])
    with open(output_dir, mode='w', encoding='utf-8') as outfile:
        for l in json_list:
            outfile.write(l.__str__())
            outfile.write('\n')
