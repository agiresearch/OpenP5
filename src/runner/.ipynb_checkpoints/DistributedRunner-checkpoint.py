from runner.SingleRunner import SingleRunner
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.distributed as dist
import logging
from tqdm import tqdm
from utils import utils
import torch
import utils.generation_trie as gt
import utils.evaluate as evaluate
from torch.utils.data.distributed import DistributedSampler
from data.TestDataset import TestDataset
from torch.utils.data import DataLoader
from processor.Collator import Collator, TestCollator
import time
import numpy as np
import random

import pdb

class DistributedRunner(SingleRunner):
    
    def __init__(self, model, tokenizer, train_loader, valid_loader, device, args, rank):
        super().__init__(model, tokenizer, train_loader, valid_loader, device, args)
        self.rank = rank
        self.model = DDP(self.model, device_ids=[self.args.gpu], find_unused_parameters=True)
        
    def train(self):
        
        self.model.zero_grad()
        train_losses = []
        valid_losses = []
        best_epoch = -1
        if self.test_before_train > 0:
            self.test()
        
        for epoch in range(self.args.epochs):
            if self.rank == 0:
                logging.info(f"Start training for epoch {epoch+1}")
                
            dist.barrier()
            if self.regenerate_candidate:
                for ds in self.train_loader.dataset.datasets:
                    ds.generate_candidates()
                    ds.construct_sentence()
            elif self.reconstruct_data:
                for ds in self.train_loader.dataset.datasets:
                    ds.construct_sentence()
                    
            self.train_loader.sampler.set_epoch(epoch)
            dist.barrier()
            
            self.model.train()
            losses = []
            
            for batch in tqdm(self.train_loader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                
                output = self.model.module(
                    input_ids=input_ids,
                    whole_word_ids=whole_input_ids,
                    attention_mask=attn,
                    labels=output_ids,
                    alpha=self.args.alpha,
                    return_dict=True,
                )
                # compute loss masking padded tokens
                loss = output["loss"]
                lm_mask = output_attention != 0
                lm_mask = lm_mask.float()
                B, L = output_ids.size()
                loss = loss.view(B, L) * lm_mask
                loss = (loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

                # update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                
                dist.barrier()
                
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                
                
                dist.all_reduce(loss.detach(), op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()
                
                dist.barrier()
                
                if self.rank == 0:
                    losses.append(loss.detach())
                
            if self.rank == 0:
                train_epoch_loss = sum(losses)/len(losses)
                train_losses.append(train_epoch_loss)
                logging.info(f"The average training loss for epoch {epoch+1} is {train_epoch_loss}")
                
            
        
            if self.valid_select > 0:
                if self.rank == 0:
                    logging.info(f"Start validation for epoch {epoch+1}")
                losses = []
                self.model.eval()
                with torch.no_grad():
                    if self.args.valid_prompt_sample > 0:
                        for ds in self.valid_loader.dataset.datasets:
                            ds.construct_sentence()
                    for batch in tqdm(self.valid_loader):
                        input_ids = batch[0].to(self.device)
                        attn = batch[1].to(self.device)
                        whole_input_ids = batch[2].to(self.device)
                        output_ids = batch[3].to(self.device)
                        output_attention = batch[4].to(self.device)

                        output = self.model.module(
                            input_ids=input_ids,
                            whole_word_ids=whole_input_ids,
                            attention_mask=attn,
                            labels=output_ids,
                            alpha=self.args.alpha,
                            return_dict=True,
                        )
                        # compute loss masking padded tokens
                        loss = output["loss"]
                        lm_mask = output_attention != 0
                        lm_mask = lm_mask.float()
                        B, L = output_ids.size()
                        loss = loss.view(B, L) * lm_mask
                        loss = (loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

                        dist.barrier()

                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        loss /= dist.get_world_size()

                        dist.barrier()

                        if self.rank == 0:
                            losses.append(loss)

                    if self.rank == 0:
                        valid_epoch_loss = sum(losses)/len(losses)
                        valid_losses.append(valid_epoch_loss)
                        logging.info(f"The average valid loss for epoch {epoch+1} is {valid_epoch_loss}")

                        if valid_epoch_loss == min(valid_losses):
                            logging.info(f"The minimal validation loss so far.")
                            best_epoch = epoch + 1
                            torch.save(self.model.module.state_dict(), self.args.model_path)
                            logging.info(f"Save the current model to {self.args.model_path}")
                
            if self.test_epoch > 0:
                if (epoch + 1) % self.test_epoch == 0:
                    self.model.eval()
                    self.test()
            
            dist.barrier()
        if self.valid_select > 0:
            if self.rank == 0:
                logging.info(f"The best validation at Epoch {best_epoch}")
        else:
            if self.rank == 0:
                torch.save(self.model.module.state_dict(), self.args.model_path)
                logging.info(f"Save the current model to {self.args.model_path}")
        
        return
    
    def get_testloader(self):
        self.testloaders = []
        datasets = self.args.datasets.split(',')
        tasks = self.args.tasks.split(',')
        if self.test_filtered > 0:
            collator = TestCollator(self.tokenizer)
        else:
            collator = Collator(self.tokenizer)
        for dataset in datasets:
            for task in tasks:
                
                testdata = TestDataset(self.args, dataset, task)
                test_sampler = DistributedSampler(testdata)
                testloader = DataLoader(dataset=testdata, sampler=test_sampler, batch_size=self.args.eval_batch_size, collate_fn=collator, shuffle=False)
                self.testloaders.append(testloader)
                
    def test(self, path=None):
        self.model.eval()
        if path:
            self.model.module.load_state_dict(torch.load(path, map_location=self.device))
        for loader in self.testloaders:
            if self.test_filtered > 0:
                if self.test_filtered_batch > 0:
                    self.test_dataset_task_filtered_batch(loader)
                else:
                    assert self.args.eval_batch_size == 1
                    self.test_dataset_task_filtered(loader)
            else:
                self.test_dataset_task(loader)
    
    def test_dataset_task_filtered_batch(self, testloader):
        if self.rank == 0:
            logging.info(f'testing filtered {testloader.dataset.dataset} dataset on {testloader.dataset.task} task')
        test_total = 0
        with torch.no_grad():
            candidates = set(testloader.dataset.all_items)
            candidate_trie = gt.Trie(
                [
                    [0] + self.tokenizer.encode(f"{testloader.dataset.dataset} item_{candidate}")
                    for candidate in candidates
                ]
                )
            prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)
            
            metrics_res = np.array([0.0] * len(self.metrics))
            for batch in tqdm(testloader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                user_idx = batch[5]
                
                
                
                prediction = self.model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=30,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=self.generate_num + testloader.dataset.max_positive,
                        num_return_sequences=self.generate_num + testloader.dataset.max_positive,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                
                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]
                
                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )
                
                rel_results = evaluate.rel_results_filtered(testloader.dataset.positive, testloader.dataset.id2user, user_idx.detach().cpu().numpy(), \
                                                            self.generate_num+testloader.dataset.max_positive, \
                                                            generated_sents, gold_sents, prediction_scores, self.generate_num)
                
                test_total += len(rel_results)
                
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)
                
            dist.barrier()
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            dist.all_reduce(metrics_res, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_total, op=dist.ReduceOp.SUM)
            
            metrics_res /= test_total
            
            if self.rank == 0:
                for i in range(len(self.metrics)):
                    logging.info(f'{self.metrics[i]}: {metrics_res[i]}')
    
    def test_dataset_task_filtered(self, testloader):
        if self.rank == 0:
            logging.info(f'testing filtered {testloader.dataset.dataset} dataset on {testloader.dataset.task} task')
        test_total = 0
        with torch.no_grad():
            candidates = set(testloader.dataset.all_items)
            
            
            metrics_res = np.array([0.0] * len(self.metrics))
            for batch in tqdm(testloader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                user_idx = int(batch[5][0])
                positive = testloader.dataset.positive[testloader.dataset.id2user[user_idx]]
                
                user_candidate = candidates - positive
                
                candidate_trie = gt.Trie(
                [
                    [0] + self.tokenizer.encode(f"{testloader.dataset.dataset} item_{candidate}")
                    for candidate in user_candidate
                ]
                )
                prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)
                
                prediction = self.model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=30,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=self.generate_num,
                        num_return_sequences=self.generate_num,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                
                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]
                
                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )
                
                rel_results = evaluate.rel_results(generated_sents, gold_sents, prediction_scores, self.generate_num)
                
                test_total += len(rel_results)
                
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)
                
            dist.barrier()
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            dist.all_reduce(metrics_res, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_total, op=dist.ReduceOp.SUM)
            
            metrics_res /= test_total
            
            if self.rank == 0:
                for i in range(len(self.metrics)):
                    logging.info(f'{self.metrics[i]}: {metrics_res[i]}')
            
    def test_dataset_task(self, testloader):
        if self.rank == 0:
            logging.info(f'testing {testloader.dataset.dataset} dataset on {testloader.dataset.task} task')
        test_total = 0
        with torch.no_grad():
            candidates = testloader.dataset.all_items
            candidate_trie = gt.Trie(
                [
                    [0] + self.tokenizer.encode(f"{testloader.dataset.dataset} item_{candidate}")
                    for candidate in candidates
                ]
            )
            prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)
            
            metrics_res = np.array([0.0] * len(self.metrics))
            for batch in tqdm(testloader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                
                prediction = self.model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=50,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=self.generate_num,
                        num_return_sequences=self.generate_num,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                
                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]
                
                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )
                
                rel_results = evaluate.rel_results(generated_sents, gold_sents, prediction_scores, self.generate_num)
                
                test_total += len(rel_results)
                
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)
                
            dist.barrier()
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            dist.all_reduce(metrics_res, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_total, op=dist.ReduceOp.SUM)
            
            metrics_res /= test_total
            
            if self.rank == 0:
                for i in range(len(self.metrics)):
                    logging.info(f'{self.metrics[i]}: {metrics_res[i]}')
                    
