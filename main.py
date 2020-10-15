from automate_multitask_bert_entity_classifier_experiments import run_auto

tasks = ["tested_positive", "tested_negative", "can_not_test", "death", "cure"]
models = ['bert-base-cased', 'digitalepidemiologylab/covid-twitter-bert']
run_auto(models[0], tasks[2], n_epochs=0, POSSIBLE_BATCH_SIZE=8, use_gpu_if_possible=False, TRAIN_RATIO=0.6,
         lr=2e-5)
