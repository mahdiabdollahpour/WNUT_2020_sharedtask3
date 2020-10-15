from auto_test import run_autotest

tasks = ["tested_positive", "tested_negative", "can_not_test", "death", "cure"]
run_autotest(tasks[2], POSSIBLE_BATCH_SIZE=8, use_gpu_if_possible=False)
