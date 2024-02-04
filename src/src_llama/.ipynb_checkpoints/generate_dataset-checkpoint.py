import argparse
import random
import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils import load_prompt_template, get_info_from_prompt,check_task_prompt
from utils import utils
from utils import indexing
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import json
import pdb

def main(args):
    tasks = args.tasks.split(',')
    
    file_data = dict()
    file_data['arguments'] = vars(args)
    file_data['data'] = []
    
    user_sequence = utils.ReadLineFromFile(os.path.join(args.data_path, args.dataset, 'user_sequence.txt'))
    user_sequence_dict = indexing.construct_user_sequence_dict(user_sequence)
    
    if args.item_indexing == 'sequential':
        print("Reindex data with sequential indexing method")
        reindex_user_seq_dict, item_map = indexing.sequential_indexing(args.data_path, args.dataset, user_sequence_dict, args.sequential_order)
    elif args.item_indexing == 'random':
        print("Reindex data with random indexing method")
        reindex_user_seq_dict, item_map = indexing.random_indexing(args.data_path, args.dataset, user_sequence_dict)
    elif args.item_indexing == 'collaborative':
        print(f"Reindex data with collaborative indexing method with token_size {args.collaborative_token_size} and {args.collaborative_cluster} cluster")
        reindex_user_seq_dict, item_map = indexing.collaborative_indexing(args.data_path, args.dataset, user_sequence_dict, \
                                                                                    args.collaborative_token_size, args.collaborative_cluster, \
                                                                                    args.collaborative_last_token, args.collaborative_float32)
    else:
        raise NotImplementedError
        
    
    # get prompt
    prompt = load_prompt_template(args.prompt_file, tasks)
    info = get_info_from_prompt(prompt)
    check_task_prompt(prompt, tasks)
    print(f"get prompt from {args.prompt_file}")
    
    
    # Load training data samples
    training_data_samples = []
    for user in reindex_user_seq_dict:
        items = reindex_user_seq_dict[user][:-2]
        for i in range(len(items)):
            if i == 0:
                if args.skip_empty_his > 0:
                    continue
            one_sample = dict()
            one_sample['dataset'] = args.dataset
            one_sample['user_id'] = user
            if args.his_prefix > 0:
                one_sample['target'] = 'item_' + items[i]
            else:
                one_sample['target'] = items[i]
            if 'history' in info:
                history = items[:i]
                if args.max_his > 0:
                    history = history[-args.max_his:]
                if args.his_prefix > 0:
                    one_sample['history'] = args.his_sep.join(["item_" + item_idx for item_idx in history])
                else:
                    one_sample['history'] = args.his_sep.join(history)
            training_data_samples.append(one_sample)
    print("load training data")
    print(f'there are {len(training_data_samples)} samples in {args.dataset} training data.')

    # construct sentences
    for i in range(len(training_data_samples)):
        one_sample = training_data_samples[i]
        for task in tasks:
            datapoint = {}
            datapoint['task'] = args.dataset + " " + task
            datapoint['data_id'] = i
            for pid in prompt[task]['seen']:
                datapoint['instruction'] = prompt[task]['seen'][pid]['Input']
                datapoint['input'] = prompt[task]['seen'][pid]['Input'].format(**one_sample)
                datapoint['output'] = prompt[task]['seen'][pid]['Output'].format(**one_sample)
                file_data['data'].append(datapoint.copy())
    
    print("data constructed")
    print(f"there are {len(file_data['data'])} prompts in {args.dataset} training data.")
    
    
    # save the data to json file
    if args.item_indexing == 'collaborative':
        output_path = f'{args.dataset}_{args.tasks}_{args.item_indexing}_{args.collaborative_token_size}_{args.collaborative_cluster}_{args.collaborative_last_token}_train.json'
    else:
        output_path = f'{args.dataset}_{args.tasks}_{args.item_indexing}_train.json'
    
    with open(os.path.join(args.data_path, args.dataset, output_path), 'w') as openfile:
        json.dump(file_data, openfile)
            
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OpenP5Dataset')
    
    # arguments related to dataset
    parser.add_argument("--data_path", type=str, default='../data', help="data directory")
    parser.add_argument("--item_indexing", type=str, default='sequential', help="item indexing method, including random, sequential and collaborative")
    parser.add_argument("--tasks", type=str, default='sequential,straightforward', help="Downstream tasks, separate by comma")
    parser.add_argument("--dataset", type=str, default='Beauty', help="Dataset name")
    parser.add_argument("--prompt_file", type=str, default='../prompt.txt', help='the path of the prompt template file')
    
    # arguments related to item indexing
    parser.add_argument("--sequential_order", type=str, default='original', help='The rank of user history during indexing')
    parser.add_argument("--collaborative_token_size", type=int, default=500, help='the number of tokens used for indexing')
    parser.add_argument("--collaborative_cluster", type=int, default=20, help='the number of clusters in each level for collaborative indexing.')
    parser.add_argument("--collaborative_last_token", type=str, default='sequential', help='how to assign the last token to items within the same clusters, random or sequential')
    parser.add_argument("--collaborative_float32", type=int, default=0, help='1 for use float32 during indexing, 0 for float64.')
    
    # arguments related to sequential task
    parser.add_argument("--max_his", type=int, default=10, help='the max number of items in history sequence, -1 means no limit')
    parser.add_argument("--his_prefix", type=int, default=1, help='whether add prefix in history')
    parser.add_argument("--his_sep", type=str, default=' , ', help='The separator used for history')
    parser.add_argument("--skip_empty_his", type=int, default=1, help='whether include data with empty history.')
    
    # 
    
    args, extras = parser.parse_known_args()
    main(args)