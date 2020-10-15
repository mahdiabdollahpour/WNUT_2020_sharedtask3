### format checking

import json
import glob


def read_json_line(path):

    output = []
    with open(path, 'r',encoding='utf-8') as f:

        lines = f.readlines()
        print(lines)
        for line in lines:
            print(line)
            jsn = json.loads(line)
            # print(jsn.)
            output.append(jsn)

    return output


def format_checker_each_file(category, input_data_path):

    print('[I] Checking', category.upper(), 'category')

    ## check file format
    input_data_path=input_data_path.replace('\\','/')
    print(input_data_path)
    try:
        input_data = read_json_line(input_data_path)
    except:
        input_data = None
        print('[ERROR] check your file format, should be .jsonl')

    # check the number of tweets
    assert len(input_data) == 500, 'check the number of predictions, should be 500'

    ## check prediction format
    for each_line in input_data:
        curr_keys = each_line.keys()
        ## first check the keys
        assert 'id' in curr_keys, 'input missing id field'
        assert 'predicted_annotation' in curr_keys, 'input missing predicted annotations'
        ## then check if all predictions are stored in list format
        for each_pred in each_line['predicted_annotation'].items():
            assert isinstance(each_pred[1], list), \
                each_pred[0] + ' contains prediction with no list format'
        ## finally check the number of keys
        if category == 'positive':
            assert len(each_line['predicted_annotation']) == 9, 'check number of slots'
        if category == 'negative':
            assert len(each_line['predicted_annotation']) == 7, 'check number of slots'
        if category == 'can_not_test':
            assert len(each_line['predicted_annotation']) == 5, 'check number of slots'
        if category == 'death':
            assert len(each_line['predicted_annotation']) == 5, 'check number of slots'
        if category == 'cure':
            assert len(each_line['predicted_annotation']) == 3, 'check number of slots'
    print('[I] You have passed the format checker for', category.upper(), 'category')

    return None


def format_checker(input_folder_path):
    ## check number of files
    input_files = glob.glob(input_folder_path+'*.jsonl')
    assert len(input_files) == 5, 'missing prediction files - should be 5 files'
    print(input_files)
    ## for each file, call format checker
    for each_file in input_files:
        # current category name
        curr_category_name = each_file.split('/')[-1].split('-')[-1].replace('.jsonl', '')
        assert curr_category_name in ['positive', 'negative', 'can_not_test', 'death',
                                      'cure'], 'check your event category name.'
        # send the file to format checker
        format_checker_each_file(curr_category_name, each_file)

    return None

format_checker('result_check/')