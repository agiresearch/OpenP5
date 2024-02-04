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
    
    
    # Load data samples
    if args.mode == 'validation':
        data_samples = load_validation(args, reindex_user_seq_dict, info)
        prompt_info = args.prompt.split(':')
        if args.item_indexing == 'collaborative':
            output_path = f'{args.dataset}_{args.tasks}_{args.item_indexing}_{args.collaborative_token_size}_{args.collaborative_cluster}_{args.collaborative_last_token}_validation_{args.prompt}.json'
        else:
            output_path = f'{args.dataset}_{args.tasks}_{args.item_indexing}_validation_{args.prompt}.json'
    elif args.mode == 'test':
        data_samples = load_test(args, reindex_user_seq_dict, info)
        prompt_info = args.prompt.split(':')
        if args.item_indexing == 'collaborative':
            output_path = f'{args.dataset}_{args.tasks}_{args.item_indexing}_{args.collaborative_token_size}_{args.collaborative_cluster}_{args.collaborative_last_token}_test_{args.prompt}.json'
        else:
            output_path = f'{args.dataset}_{args.tasks}_{args.item_indexing}_test_{args.prompt}.json'
    else:
        raise NotImplementedError
    print(f'there are {len(data_samples)} samples in {args.dataset} {args.mode} data.')
    print(prompt_info)

    # construct sentences
    for i in range(len(data_samples)):
        one_sample = data_samples[i]
        for task in tasks:
            datapoint = {}
            datapoint['task'] = args.dataset + " " + task
            datapoint['instruction'] = prompt[task][prompt_info[0]][prompt_info[1]]['Input']
            datapoint['input'] = prompt[task][prompt_info[0]][prompt_info[1]]['Input'].format(**one_sample)
            datapoint['output'] = prompt[task][prompt_info[0]][prompt_info[1]]['Output'].format(**one_sample)
            file_data['data'].append(datapoint.copy())
    
    print("data constructed")
    print(f"there are {len(file_data['data'])} prompts in {args.dataset} {args.mode} data.")
    
    
    # save the data to json file
    
    with open(os.path.join(args.data_path, args.dataset, output_path), 'w') as openfile:
        json.dump(file_data, openfile)
            
    
def load_test(args, reindex_user_seq_dict, info):
    data_samples = []
    for user in reindex_user_seq_dict:
        items = reindex_user_seq_dict[user]
        one_sample = dict()
        one_sample['dataset'] = args.dataset
        one_sample['user_id'] = user
        if args.his_prefix > 0:
            one_sample['target'] = 'item_' + items[-1]
        else:
            one_sample['target'] = items[-1]
        if 'history' in info:
            history = items[:-1]
            if args.max_his > 0:
                history = history[-args.max_his:]
            if args.his_prefix > 0:
                one_sample['history'] = args.his_sep.join(["item_" + item_idx for item_idx in history])
            else:
                one_sample['history'] = args.his_sep.join(history)
        data_samples.append(one_sample)
    return data_samples

def load_validation(args, reindex_user_seq_dict, info):
    data_samples = []
    for user in reindex_user_seq_dict:
        items = reindex_user_seq_dict[user]
        one_sample = dict()
        one_sample['dataset'] = args.dataset
        one_sample['user_id'] = user
        if args.his_prefix > 0:
            one_sample['target'] = 'item_' + items[-2]
        else:
            one_sample['target'] = items[-2]
        if 'history' in info:
            history = items[:-2]
            if args.max_his > 0:
                history = history[-args.max_his:]
            if args.his_prefix > 0:
                one_sample['history'] = args.his_sep.join(["item_" + item_idx for item_idx in history])
            else:
                one_sample['history'] = args.his_sep.join(history)
        data_samples.append(one_sample)
    return data_samples
    
    
    
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
    parser.add_argument("--collaborative_token_size", type=int, default=100, help='the number of tokens used for indexing')
    parser.add_argument("--collaborative_cluster", type=int, default=20, help='the number of clusters in each level for collaborative indexing.')
    parser.add_argument("--collaborative_last_token", type=str, default='sequential', help='how to assign the last token to items within the same clusters, random or sequential')
    parser.add_argument("--collaborative_float32", type=int, default=0, help='1 for use float32 during indexing, 0 for float64.')
    
    # arguments related to sequential task
    parser.add_argument("--max_his", type=int, default=10, help='the max number of items in history sequence, -1 means no limit')
    parser.add_argument("--his_prefix", type=int, default=1, help='whether add prefix in history')
    parser.add_argument("--his_sep", type=str, default=' , ', help='The separator used for history')
    parser.add_argument("--skip_empty_his", type=int, default=1, help='whether include data with empty history.')
    
    parser.add_argument("--mode", type=str, default='validation', help='validation or test')
    parser.add_argument("--prompt", type=str, default='seen:0', help='The prompt used for evaluation, seen/unseen: id')
    
    args, extras = parser.parse_known_args()
    main(args)