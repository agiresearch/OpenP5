import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
from tqdm import tqdm
from utils import utils
import utils.generation_trie as gt
from data.TestDataset import TestDataset
from torch.utils.data import DataLoader
from processor.Collator import Collator
import time
import pdb

class SingleRunner:
    def parse_runner_args(parser):
        """
        parse dataset related command line arguments
        """
        parser.add_argument("--optim", type=str, default='AdamW', help='The name of the optimizer')
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--clip", type=float, default=1)
        parser.add_argument("--logging_step", type=int, default=100)
        parser.add_argument("--warmup_prop", type=float, default=0.05)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--adam_eps", type=float, default=1e-6)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--alpha", type=float, default=2)
        parser.add_argument("--train", type=int, default=1, help='train or not')
        parser.add_argument("--backbone", type=str, default='t5-small', help='backbone model name')
        parser.add_argument("--metrics", type=str, default='hit@5,hit@10,ndcg@5,ndcg@10', help='Metrics used for evaluation')
        parser.add_argument("--load", type=int, default=0, help='load model from model path or not.')
        parser.add_argument("--random_initialize", type=int, default=1, help='Randomly initialize number-related tokens.')
        parser.add_argument("--test_epoch", type=int, default=1, help='test once for how many epochs, 0 for no test during training.')
        parser.add_argument("--valid_select", type=int, default=0, help='use validation loss to select models')
        parser.add_argument("--test_before_train", type=int, default=1, help='whether test before training')
        
        return parser
    
    def __init__(self, model, tokenizer, train_loader, valid_loader, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.args = args
        self.regenerate_candidate = 'candidate_items' in self.train_loader.dataset.datasets[0].info
        self.reconstruct_data = self.args.sample_prompt
        self.test_epoch = self.args.test_epoch
        self.valid_select = self.args.valid_select
        self.test_before_train = self.args.test_before_train
        
        self.get_testloader()
        
        if args.train:
            self.optimizer, self.scheduler = self.create_optimizer_and_scheduler()
            
        self.metrics = args.metrics.split(',')
        self.generate_num = max([int(m.split('@')[1]) for m in self.metrics])
        
        
    def train(self):
        self.model.zero_grad()
        train_losses = []
        eval_losses = []
        best_epoch = -1
        
        if self.test_before_train > 0:
            self.test()
        for epoch in range(self.args.epochs):
            if self.regenerate_candidate:
                for ds in self.train_loader.dataset.datasets:
                    ds.generate_candidates()
                    ds.construct_sentence()
            elif self.reconstruct_data:
                for ds in self.train_loader.dataset.datasets:
                    ds.construct_sentence()
            self.train_loader.sampler.set_epoch(epoch)
            logging.info(f"Start training for epoch {epoch+1}")
            self.model.train()
            losses = []
            for batch in tqdm(self.train_loader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                
                output = self.model(
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
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                losses.append(loss.detach())
            train_epoch_loss = sum(losses)/len(losses)
            train_losses.append(train_epoch_loss)
            logging.info(f"The average training loss for epoch {epoch+1} is {train_epoch_loss}")
            
            self.test()
            
            if self.valid_select > 0:
                logging.info(f"Start validation for epoch {epoch+1}")
                losses = []
                self.model.eval()
                with torch.no_grad():
                    for batch in tqdm(self.valid_loader):
                        input_ids = batch[0].to(self.device)
                        attn = batch[1].to(self.device)
                        whole_input_ids = batch[2].to(self.device)
                        output_ids = batch[3].to(self.device)
                        output_attention = batch[4].to(self.device)

                        output = self.model(
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

                        losses.append(loss.detach())
                    valid_epoch_loss = sum(losses)/len(losses)
                    valid_losses.append(valid_epoch_loss)
                    logging.info(f"The average valid loss for epoch {epoch+1} is {valid_epoch_loss}")

                    if valid_epoch_loss == min(valid_losses):
                        logging.info(f"The minimal validation loss so far.")
                        best_epoch = epoch + 1
                        utils.save_model(self.model, self.args.model_path)
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
    
    def create_optimizer_and_scheduler(self):
        if self.args.rank == 0:
            logging.info("Building Optimizer and Scheduler")
        batch_per_epoch = len(self.train_loader)
        total_steps = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epochs
        warmup_steps = int(total_steps * self.args.warmup_prop)
        
        if self.args.rank == 0:
            logging.info(f'Batch per epoch: {batch_per_epoch}')
            logging.info(f'Total steps: {total_steps}')
            logging.info(f'Warmup proportion: {self.args.warmup_prop}')
            logging.info(f'Warm up steps: {warmup_steps}')

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.args.rank == 0:
            logging.info(f"Building Optimizer {self.args.optim}")
        
        if self.args.optim.lower() == 'adamw':
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_eps)
        else:
            raise NotImplementError
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        return optimizer, scheduler
    
    def get_testloader(self):
        self.testloaders = []
        datasets = self.args.datasets.split(',')
        tasks = self.args.tasks.split(',')
        collator = Collator(self.tokenizer)
        for dataset in datasets:
            for task in tasks:
                testdata = TestDataset(self.args, dataset, task)
                testloader = DataLoader(dataset=testdata, batch_size=self.args.eval_batch_size, collate_fn=collator, shuffle=False)
                self.testloaders.append(testloader)
        
    def test(self, path=None):
        self.model.eval()
        if path:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        for loader in self.testloaders:
            self.test_dataset_task(loader)
                
        
        
    def test_dataset_task(self, testloader):
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
                        max_length=8,
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
                
                # print(generated_sents)
                # exit()
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
