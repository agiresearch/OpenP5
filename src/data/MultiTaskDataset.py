import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.prompt import load_prompt_template, get_info_from_prompt,check_task_prompt
from utils import utils
from utils import indexing
from collections import defaultdict
import logging
import re
import pdb


class MultiTaskDataset(Dataset):
    def parse_dataset_args(parser):
        """
        parse dataset related command line arguments
        """
        parser.add_argument("--data_path", type=str, default='../data', help="data directory")
        parser.add_argument("--item_indexing", type=str, default='sequential', help="item indexing method, including random, sequential and collaborative")
        parser.add_argument("--tasks", type=str, default='sequential,direct,straightforward', help="Downstream tasks, separate by comma")
        parser.add_argument("--datasets", type=str, default='Beauty', help="Dataset names, separate by comma")
        parser.add_argument("--prompt_file", type=str, default='../prompt_template.txt', help='the path of the prompt template file')
        
        
        # arguments related to item indexing methods
        parser.add_argument("--sequential_order", type=str, default='original', help='The rank of user history during ')
        parser.add_argument("--collaborative_token_size", type=int, default=200, help='the number of tokens used for indexing')
        parser.add_argument("--collaborative_cluster", type=int, default=20, help='the number of clusters in each level for collaborative indexing.')
        parser.add_argument("--collaborative_last_token", type=str, default='sequential', help='how to assign the last token to items within the same clusters, random or sequential')
        
        # arguments related to sequential task
        parser.add_argument("--max_his", type=int, default=-1, help='the max number of items in history sequence, -1 means no limit')
        parser.add_argument("--his_prefix", type=int, default=1, help='whether add prefix in history')
        parser.add_argument("--his_sep", type=str, default=' , ', help='The separator used for history')
        parser.add_argument("--skip_empty_his", type=int, default=1, help='whether include data with empty history.')
        
        # arguments related to direct task
        parser.add_argument("--candidate_neg_num", type=int, default=100, help='the number of negative candidate itmes in direct task.')
        parser.add_argument("--candidate_sep", type=str, default=', ', help='The separator used for candidate items')
        
        # arguments related for evaluation
        parser.add_argument("--valid_prompt", type=str, default='seen:0', help='The prompt used for evaluation, seen/unseen: id')
        parser.add_argument("--valid_prompt_sample", type=int, default=1, help='use sampled prompt for validation every epoch.')
        parser.add_argument("--valid_sample_num", type=str, default='3,3', help='the number of sampled data for each task')
        parser.add_argument("--test_prompt", type=str, default='seen:0', help='The prompt used for evaluation, seen/unseen: id')
        
        # arguments related to prompt sampling
        parser.add_argument("--sample_prompt", type=int, default=0, help='sample prompt or not')
        parser.add_argument("--sample_num", type=str, default='2,2,2', help='the number of sampled data for each task')
        
        return parser
        
    def __init__(self, args, dataset, mode):
        super().__init__()
        self.data_path = args.data_path
        self.dataset = dataset
        self.tasks = args.tasks.split(',')
        if args.sample_prompt > 0:
            assert len(self.tasks) == len(args.sample_num.split(',')), "prompt sample number does not match task number"
        self.item_indexing = args.item_indexing
        self.mode = mode
        self.args = args
        
        self.rank = args.rank
        self.prefix = args.his_prefix
        self.skip_empty_his = args.skip_empty_his
        self.collaborative_token_size = self.args.collaborative_token_size
        self.collaborative_cluster_num = self.args.collaborative_cluster
        self.collaborative_last_token = self.args.collaborative_last_token
        
        if self.rank == 0:
            logging.info(f"Generating data for {self.dataset} dataset")
        
        # load and check prompt
        if self.rank == 0:
            logging.info(f"Get prompt template from {args.prompt_file}")
        self.prompt = load_prompt_template(args.prompt_file, self.tasks)
        if self.rank == 0:
            logging.info(f"{self.prompt['sequential']['seen']['0']['Input']}")
        check_task_prompt(self.prompt, self.tasks)
        self.info = get_info_from_prompt(self.prompt)
        if self.rank == 0:
            logging.info(f"Required info: {self.info}")
        
        if 'history' in self.info:
            self.max_his = args.max_his
            self.his_sep = args.his_sep
        if 'candidate_items' in self.info:
            self.candidate_neg_num = args.candidate_neg_num
            self.candidate_sep = args.candidate_sep
        
        # load user sequence data
        self.user_sequence = utils.ReadLineFromFile(os.path.join(self.data_path, self.dataset, 'user_sequence.txt'))
        self.user_sequence_dict = indexing.construct_user_sequence_dict(self.user_sequence)
        
        # apply indexing method
        if self.item_indexing == 'sequential':
            if self.rank == 0:
                logging.info("Reindex data with sequential indexing method")
            self.reindex_user_seq_dict, self.item_map = indexing.sequential_indexing(self.data_path, self.dataset, self.user_sequence_dict, args.sequential_order)
        elif self.item_indexing == 'random':
            if self.rank == 0:
                logging.info("Reindex data with random indexing method")
            self.reindex_user_seq_dict, self.item_map = indexing.random_indexing(self.data_path, self.dataset, self.user_sequence_dict)
        elif self.item_indexing == 'collaborative':
            if self.rank == 0:
                logging.info(f"Reindex data with collaborative indexing method with token_size {self.collaborative_token_size} and {self.collaborative_cluster_num} cluster")
            self.reindex_user_seq_dict, self.item_map = indexing.collaborative_indexing(self.data_path, self.dataset, self.user_sequence_dict, \
                                                                                        self.collaborative_token_size, self.collaborative_cluster_num, self.collaborative_last_token)
            self.new_token = []
            for idx in list(self.item_map.values()):
                self.new_token += re.findall(r'\<.*?\>', idx)
        else:
            raise NotImplementedError
            
            
        self.all_items = list(self.item_map.values())
            
        # get positive samples for each user to sample negative candidates or evaluation
        self.positive = self.get_positive()
        
        # load data
        if self.mode == 'train':
            if self.rank == 0:
                logging.info("loading training data")
            self.data_samples = self.load_train()
        elif self.mode == 'validation':
            self.data_samples = self.load_validation()
            if self.rank == 0:
                logging.info("loading validation data")
            self.valid_prompt = args.valid_prompt
            if self.rank == 0:
                logging.info(f"The validation prompt is {self.valid_prompt}")
        else:
            raise NotImplementedError
            
        # sample candidate items
        if 'candidate_items' in self.info:
            if self.rank == 0:
                logging.info(f"Generating candidates for {self.mode} in {self.dataset} dataset")
            self.generate_candidates()
            
        # get prompt related info, including numbers and index
        self.get_prompt_info()
        
        self.construct_sentence()
    
    def get_positive(self):
        """
        Get a dict of set to save the positive interactions for negative candidate sampling
        """
        positive = dict()
        for user in self.reindex_user_seq_dict:
            if self.mode == 'train':
                positive[user] = set(self.reindex_user_seq_dict[user][:-2])
            if self.mode == 'validation':
                positive[user] = set(self.reindex_user_seq_dict[user][:-1])
            if self.mode == 'test':
                positive[user] = set(self.reindex_user_seq_dict[user])
        return positive
    
    def shuffle(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        
        for task in self.task_data:
            indices = torch.randperm(len(self.task_data[task]), generator=g).tolist()
            self.task_data[task] = [self.task_data[task][i] for i in indices]
        
        
    def get_prompt_info(self):
        """
        Calculate number of prompts and cumulative index for each task
        - task_prompt_num: save the number of prompts for each task
        - task_index: the cumulative index for each task. if task_index[i-1] <= idx < task_index[i], then the idx belongs to task[i]
            - For example, there are 100 data samples in total, there are 3 tasks, the task_prompt_num is [2,1,3], then the task_index is [200, 300, 600].
        """
        if self.rank == 0:
            logging.info(f"Getting prompt information")
        if self.mode == 'train':
            if self.args.sample_prompt == 0:
                self.task_prompt_num = [len(self.prompt[task]['seen']) for task in self.tasks]
            else:
                sample_number = self.args.sample_num.split(',')
                self.task_prompt_num = [int(sample_number[i]) for i in range(len(self.tasks))]
        else:
            if self.args.valid_prompt_sample == 0:
                self.task_prompt_num = [1] * len(self.tasks)
            else:
                sample_number = self.args.valid_sample_num.split(',')
                self.task_prompt_num = [int(sample_number[i]) for i in range(len(self.tasks))]
        self.task_index = [self.task_prompt_num[0] * len(self.data_samples)]
        for i in range(1, len(self.task_prompt_num)):
            self.task_index.append(self.task_index[i-1] + self.task_prompt_num[i] * len(self.data_samples))
        self.task_data = dict()
        for i in range(len(self.tasks)):
            if i == 0:
                start = 0
            else:
                start = self.task_index[i-1]
            end = self.task_index[i]
            task = self.tasks[i]
            self.task_data[task] = [i for i in range(start, end)]
    
    def load_train(self):
        """
        Load training data samples
        """
        data_samples = []
        for user in self.reindex_user_seq_dict:
            items = self.reindex_user_seq_dict[user][:-2]
            for i in range(len(items)):
                if i == 0:
                    if self.skip_empty_his > 0:
                        continue
                one_sample = dict()
                one_sample['dataset'] = self.dataset
                one_sample['user_id'] = user
                if self.prefix > 0:
                    one_sample['target'] = 'item_' + items[i]
                else:
                    one_sample['target'] = items[i]
                if 'history' in self.info:
                    history = items[:i]
                    if self.max_his > 0:
                        history = history[-self.max_his:]
                    if self.prefix > 0:
                        one_sample['history'] = self.his_sep.join(["item_" + item_idx for item_idx in history])
                    else:
                        one_sample['history'] = self.his_sep.join(history)
                data_samples.append(one_sample)
        return data_samples
    
    def load_validation(self):
        """
        Load validation data samples
        """
        data_samples = []
        for user in self.reindex_user_seq_dict:
            items = self.reindex_user_seq_dict[user]
            one_sample = dict()
            one_sample['dataset'] = self.dataset
            one_sample['user_id'] = user
            if self.prefix > 0:
                one_sample['target'] = 'item_' + items[-2]
            else:
                one_sample['target'] = items[-2]
            if 'history' in self.info:
                history = items[:-2]
                if self.max_his > 0:
                    history = history[-self.max_his:]
                if self.prefix > 0:
                    one_sample['history'] = self.his_sep.join(["item_" + item_idx for item_idx in history])
                else:
                    one_sample['history'] = self.his_sep.join(history)
            data_samples.append(one_sample)
        return data_samples
    
    def generate_candidates(self):
        """
        Generate candidate items for each data sample, the candidate items include the target item and negative items 
        """
        
        for i in range(len(self.data_samples)):
            row = self.data_samples[i]
            user = row['user_id']
            item = row['target']
            i = 0
            neg = []
            while i < self.candidate_neg_num:
                n = random.randint(0, len(self.all_items) - 1)
                if self.all_items[n] not in self.positive[user]:
                    neg.append(self.all_items[n])
                    i += 1
            neg.append(item)
            random.shuffle(neg)
            row['candidate_items'] = self.candidate_sep.join(neg)
        return
        
    def __len__(self):
        return len(self.data['input'])
    
    # def identify_prompt(self, idx):
    #     for i in range(len(self.tasks)):
    #         if idx < self.task_index[i]:
    #             if i == 0:
    #                 intask_id = idx
    #             else:
    #                 intask_id = idx - self.task_index[i-1]
    #             task = self.tasks[i]
    #             task_id = i
    #             break
    #     if self.mode == 'train':
    #         data_id = intask_id // self.task_prompt_num[task_id]
    #         prompt_id = intask_id % self.task_prompt_num[task_id]
    #         prompt = self.prompt[task]['seen'][str(prompt_id)]
    #     if self.mode == 'validation':
    #         data_id = intask_id
    #         info = self.valid_prompt.split(':')
    #         prompt = self.prompt[task][info[0]][info[1]]
    #     if self.mode == 'test':
    #         data_id = intask_id
    #         info = self.test_prompt.split(':')
    #         prompt = self.prompt[task][info[0]][info[1]]
    #     return data_id, prompt
    
    
    def construct_sentence(self):
        if self.mode == 'train':
            if self.args.sample_prompt == 0:
                self._construct_sentence_all()
            else:
                self._construct_sentence_sample()
            if self.rank == 0:
                logging.info(f"Input: {self.data['input'][100]} , Output: {self.data['output'][100]} ")
        elif self.mode == 'validation':
            if self.args.valid_prompt_sample == 0:
                self._construct_sentence_valid()
            else:
                self._construct_sentence_sample()
            if self.rank == 0:
                logging.info(f"Input: {self.data['input'][100]} , Output: {self.data['output'][100]} ")
                logging.info(f"Input: {self.data['input'][101]} , Output: {self.data['output'][101]} ")
    
    def _construct_sentence_valid(self):
        self.data = {}
        self.data['input'] = []
        self.data['output'] = []
        setting = self.valid_prompt.split(':')
        for task in self.tasks:
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                self.data['input'].append(self.prompt[task][setting[0]][setting[1]]['Input'].format(**datapoint))
                self.data['output'].append(self.prompt[task][setting[0]][setting[1]]['Output'].format(**datapoint))
    
    def _construct_sentence_all(self):
        self.data = {}
        self.data['input'] = []
        self.data['output'] = []
        for task in self.tasks:
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                for pid in self.prompt[task]['seen']:
                    self.data['input'].append(self.prompt[task]['seen'][pid]['Input'].format(**datapoint))
                    self.data['output'].append(self.prompt[task]['seen'][pid]['Output'].format(**datapoint))
                    
    def _construct_sentence_sample(self):
        self.data = {}
        self.data['input'] = []
        self.data['output'] = []
        for t in range(len(self.tasks)):
            task = self.tasks[t]
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                for j in range(self.task_prompt_num[t]):
                    pid = random.randint(0, len(self.prompt[task]['seen']) - 1)
                    self.data['input'].append(self.prompt[task]['seen'][str(pid)]['Input'].format(**datapoint))
                    self.data['output'].append(self.prompt[task]['seen'][str(pid)]['Output'].format(**datapoint))
        
    
    def __getitem__(self, idx):
        # data_id, prompt = self.identify_prompt(idx)
        # datapoint = self.data_samples[data_id]
        
        # return {'input': prompt['Input'].format(**datapoint),
        #        'output': prompt['Output'].format(**datapoint)}
        
        return {'input': self.data['input'][idx],
               'output': self.data['output'][idx]}