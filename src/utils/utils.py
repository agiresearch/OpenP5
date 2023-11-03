import numpy as np
import os
import pickle
import argparse
import inspect
import logging
import sys
import random
import torch
import torch.nn as nn
import re


def parse_args(parser):
    
    # global arguments
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--model_dir", type=str, default='../model', help='The model directory')
    parser.add_argument("--checkpoint_dir", type=str, default='../checkpoint', help='The checkpoint directory')
    parser.add_argument("--model_name", type=str, default='model.pt', help='The model name')
    parser.add_argument("--log_dir", type=str, default='../log', help='The log directory')
    parser.add_argument("--master_addr", type=str, default='localhost', help='Setup MASTER_ADDR for os.environ')
    parser.add_argument("--master_port", type=str, default='12345', help='Setup MASTER_PORT for os.environ')
    parser.add_argument('--logging_level', type=int, default=logging.INFO,help='Logging Level, 0, 10, ..., 50')
    
    # arguments related to dataset
    parser.add_argument("--data_path", type=str, default='../data', help="data directory")
    parser.add_argument("--item_indexing", type=str, default='sequential', help="item indexing method, including random, sequential and collaborative")
    parser.add_argument("--tasks", type=str, default='sequential,straightforward', help="Downstream tasks, separate by comma")
    parser.add_argument("--datasets", type=str, default='Beauty', help="Dataset names, separate by comma")
    parser.add_argument("--prompt_file", type=str, default='../prompt.txt', help='the path of the prompt template file')


    # arguments related to item indexing methods
    parser.add_argument("--sequential_order", type=str, default='original', help='The rank of user history during ')
    parser.add_argument("--collaborative_token_size", type=int, default=200, help='the number of tokens used for indexing')
    parser.add_argument("--collaborative_cluster", type=int, default=20, help='the number of clusters in each level for collaborative indexing.')
    parser.add_argument("--collaborative_last_token", type=str, default='sequential', help='how to assign the last token to items within the same clusters, random or sequential')
    parser.add_argument("--collaborative_float32", type=int, default=0, help='1 for use float32 during indexing, 0 for float64.')

    # arguments related to sequential task
    parser.add_argument("--max_his", type=int, default=10, help='the max number of items in history sequence, -1 means no limit')
    parser.add_argument("--his_prefix", type=int, default=1, help='whether add prefix in history')
    parser.add_argument("--his_sep", type=str, default=' , ', help='The separator used for history')
    parser.add_argument("--skip_empty_his", type=int, default=1, help='whether include data with empty history.')

    # arguments related for evaluation
    parser.add_argument("--valid_prompt", type=str, default='seen:0', help='The prompt used for evaluation, seen/unseen: id')
    parser.add_argument("--valid_prompt_sample", type=int, default=1, help='use sampled prompt for validation every epoch.')
    parser.add_argument("--valid_sample_num", type=str, default='3,3', help='the number of sampled data for each task')
    parser.add_argument("--test_prompt", type=str, default='seen:0', help='The prompt used for evaluation, seen/unseen: id')

    # arguments related to prompt
    parser.add_argument("--sample_prompt", type=int, default=0, help='sample prompt or not')
    parser.add_argument("--sample_num", type=str, default='2,2', help='the number of sampled data for each task')
    parser.add_argument("--cutoff", type=int, default=1024, help='cutoff length for data')
    
    
    # arguments related to sampler
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="the batch size for evaluation")
    parser.add_argument("--group_task_in_batch", type=int, default=1, help='Whether group data for one task in the batch. If so, use customized sampler')
    parser.add_argument("--task_alternating_optim", type=int, default=0, help='Whether use alternating optimizations')
    
    # arguments related to trainer
    parser.add_argument("--optim", type=str, default='adamw_torch', help='The name of the optimizer')
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--backbone", type=str, default='t5-small', help='backbone model name')
    parser.add_argument("--random_initialize", type=int, default=1, help='Randomly initialize number-related tokens.')
    parser.add_argument("--test_epoch", type=int, default=1, help='test once for how many epochs, 0 for no test during training.')
    parser.add_argument("--valid_select", type=int, default=0, help='use validation loss to select models')
    
    # arguments related to lora
    parser.add_argument("--lora", type=int, default=0, help='whether user lora.')
    parser.add_argument("--lora_r", type=int, default=8, help='lora parameter lora_r.')
    parser.add_argument("--lora_alpha", type=int, default=16, help='lora parameter lora_alpha.')
    parser.add_argument("--lora_dropout", type=float, default=0.05, help='lora parameter lora_dropout.')
    parser.add_argument("--lora_target_modules", type=str, default='q_proj,v_proj,embed_tokens', help='lora parameter lora_r.')
    
    # arguments related to evaluation
    parser.add_argument("--metrics", type=str, default='hit@5,hit@10,ndcg@5,ndcg@10', help='Metrics used for evaluation')
    # parser.add_argument("--test_before_train", type=int, default=1, help='whether test before training')
    # parser.add_argument("--test_filtered", type=int, default=0, help='whether filter out the items in the training data.')
    # parser.add_argument("--test_filtered_batch", type=int, default=1, help='whether testing with filtered data in batch.')
    
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False



def random_initialization(model, tokenizer, backbone):
    ids = []
    for x in range(30000):
        tokenized_ids = tokenizer.encode(str(x))
        if 3 in tokenized_ids:
            tokenized_ids.remove(3)
        if 1 in tokenized_ids:
            tokenized_ids.remove(1)
        ids += tokenized_ids
    ids = list(set(ids))
    
    # reinitialize the embedding in the backbone models
    for index in ids:
        if 't5' in backbone:
            model.shared.weight.data[index] = nn.init.normal_(
                model.shared.weight.data[index], 0, 1.0
            )
        elif 'llama' in backbone.lower():
            model.model.embed_tokens.weight.data[index] = nn.init.normal_(
                model.model.embed_tokens.weight.data[index], 0, 1.0
            )

    return model


def setup_logging(args):
    args.log_name = log_name(args)
    if len(args.datasets.split(',')) > 1:
        folder_name = 'SP5'
    else:
        folder_name = args.datasets
    folder = os.path.join(args.log_dir, folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    log_file = os.path.join(args.log_dir, folder_name, args.log_name + '.log')
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=args.logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    return


def log_name(args):
    if len(args.datasets.split(',')) > 1:
        folder_name = 'SP5'
    else:
        folder_name = args.datasets
    params = [str(args.sample_prompt), str(args.his_prefix), str(args.skip_empty_his), str(args.max_his), str(args.master_port), folder_name, args.tasks, args.backbone, args.item_indexing, str(args.lr), str(args.epochs), str(args.batch_size), args.sample_num, args.prompt_file[3:-4]]
    return '_'.join(params)


def setup_model_path(args):
    if len(args.datasets.split(',')) > 1:
        folder_name = 'SP5'
    else:
        folder_name = args.datasets
    if args.model_name == 'model.pt':
        model_path = os.path.join(args.model_dir, folder_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        args.model_path = os.path.join(model_path, args.log_name+'.pt')
    else:
        args.model_path = os.path.join(args.checkpoint_dir, args.model_name)
    return


def ReadLineFromFile(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def WriteDictToFile(path, write_dict):
    with open(path, 'w') as out:
        for user, items in write_dict.items():
            if type(items) == list:
                out.write(user + ' ' + ' '.join(items) + '\n')
            else:
                out.write(user + ' ' + str(items) + '\n')

                
def load_prompt_template(path, task_list):
    """
    Load prompt template from the file. Keep training tasks only.
    Input:
    - path: The path for prompt template txt file.
    - task_list: A list of required tasks.
    Return:
    - prompt_templates: a dictionary of prompt templates. e.g., {task: {'seen': {'0': {'Input': template_input, 'Output': template_output}}}}
    
    """
    
    if not os.path.exists(path):
        raise FileNotFoundError
    prompt_info = ReadLineFromFile(path)
    prompt_templates = dict()
    for prompt in prompt_info:
        t = [sens.strip() for sens in prompt.split(';')]
        if t[0] not in task_list:
            continue
        if t[0] not in prompt_templates:
            prompt_templates[t[0]] = dict()
        if t[1] not in prompt_templates[t[0]]:
            prompt_templates[t[0]][t[1]] = dict()
        num = len(prompt_templates[t[0]][t[1]])
        prompt_templates[t[0]][t[1]][str(num)] = dict()
        prompt_templates[t[0]][t[1]][str(num)]['Input'] = t[2]
        prompt_templates[t[0]][t[1]][str(num)]['Output'] = t[3]
    return prompt_templates

def get_info_from_prompt(prompt_templates):
    """
    Extract the require information from the prompt templates.
    Input:
    - prompt_templates: a dictionary of prompt templates.
    Output:
    - info: a list of required information.
    """
    
    info = []
    for task in prompt_templates:
        for see in prompt_templates[task]:
            for i in prompt_templates[task][see]:
                info += re.findall(r'\{.*?\}', prompt_templates[task][see][i]['Input'])
                info += re.findall(r'\{.*?\}', prompt_templates[task][see][i]['Output'])
    info = [i[1:-1] for i in set(info)]
    return info

def check_task_prompt(prompt_templates, task_list):
    """
    Check if all tasks have prompt templates. Raise Error if training tasks have no prompt.
    Input:
    - prompt_templates: A dictionary of prompt templates.
    - task_list: A list of training tasks.
    """
    for task in task_list:
        assert task in prompt_templates, f"No prompt for {task} task"
        
        
