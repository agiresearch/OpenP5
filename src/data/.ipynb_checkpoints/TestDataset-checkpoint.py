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
import pdb
import re


class TestDataset(Dataset):
    def __init__(self, args, dataset, task):
        super().__init__()
        self.data_path = args.data_path
        self.dataset = dataset
        self.task = task
        self.item_indexing = args.item_indexing
        
        self.collaborative_token_size = args.collaborative_token_size
        self.collaborative_cluster_num = args.collaborative_cluster
        
        self.prompt = load_prompt_template(args.prompt_file, [self.task])
        check_task_prompt(self.prompt, [self.task])
        self.info = get_info_from_prompt(self.prompt)
        
        if 'history' in self.info:
            self.max_his = args.max_his
            self.his_sep = args.his_sep
        if 'candidate_items' in self.info:
            self.candidate_neg_num = args.candidate_neg_num
            self.candidate_sep = args.candidate_sep
        
        # load user sequence data
        self.user_sequence = utils.ReadLineFromFile(os.path.join(self.data_path, self.dataset, 'user_sequence.txt'))
        self.user_sequence_dict = indexing.construct_user_sequence_dict(self.user_sequence)
        
        self.prefix = args.his_prefix
        
        # apply indexing method
        if self.item_indexing == 'sequential':
            self.reindex_user_seq_dict, self.item_map = indexing.sequential_indexing(self.data_path, self.dataset, self.user_sequence_dict, args.sequential_order)
        elif self.item_indexing == 'random':
            self.reindex_user_seq_dict, self.item_map = indexing.random_indexing(self.data_path, self.dataset, self.user_sequence_dict)
        elif self.item_indexing == 'collaborative':
            self.reindex_user_seq_dict, self.item_map = indexing.collaborative_indexing(self.data_path, self.dataset, self.user_sequence_dict, \
                                                                                        self.collaborative_token_size, self.collaborative_cluster_num)
            self.new_token = []
            for idx in list(self.item_map.values()):
                self.new_token += re.findall(r'\<.*?\>', idx)
        else:
            raise NotImplementedError
            
        self.all_items = list(self.item_map.values())
            
        # get positive samples for each user to sample negative candidates or evaluation
        self.positive = self.get_positive()
        
        # load data
        self.data_samples = self.load_test()
        self.test_prompt = args.test_prompt
        
            
        # sample candidate items
        if 'candidate_items' in self.info:
            self.generate_candidates()
            
        self.construct_sentence()
        # get prompt related info, including numbers and index
        # self.get_prompt_info()
        
    def load_test(self):
        """
        Load test data samples
        """
        data_samples = []
        for user in self.reindex_user_seq_dict:
            items = self.reindex_user_seq_dict[user]
            one_sample = dict()
            one_sample['dataset'] = self.dataset
            one_sample['user_id'] = user
            if self.prefix > 0:
                one_sample['target'] = 'item_' + items[-1]
            else:
                one_sample['target'] = items[-1]
            if 'history' in self.info:
                history = items[:-1]
                if self.max_his > 0:
                    history = history[-self.max_his:]
                if self.prefix > 0:
                    one_sample['history'] = self.his_sep.join(["item_" + item_idx for item_idx in history])
                else:
                    one_sample['history'] = self.his_sep.join(history)
            data_samples.append(one_sample)
        return data_samples
    
    def get_positive(self):
        """
        Get a dict of set to save the positive interactions for negative candidate sampling
        """
        positive = dict()
        for user in self.reindex_user_seq_dict:
            positive[user] = set(self.reindex_user_seq_dict[user])
        return positive
    
    def __len__(self):
        return len(self.data_samples)
    
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
    
    def construct_sentence(self):
        self.data = {}
        self.data['input'] = []
        self.data['output'] = []
        info = self.test_prompt.split(':')
        prompt = self.prompt[self.task][info[0]][info[1]]
        for i in range(len(self.data_samples)):
            datapoint = self.data_samples[i]
            self.data['input'].append(prompt['Input'].format(**datapoint))
            self.data['output'].append(prompt['Output'].format(**datapoint))
        
    
    def __getitem__(self, idx):
        # data_id, prompt = self.identify_prompt(idx)
        # datapoint = self.data_samples[data_id]
        
        # return {'input': prompt['Input'].format(**datapoint),
        #        'output': prompt['Output'].format(**datapoint)}
        
        return {'input': self.data['input'][idx],
               'output': self.data['output'][idx]}
    
    