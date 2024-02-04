from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader
import torch
import inspect
import transformers
from transformers.utils import find_labels
import random
import math

class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """
    def __init__(self, model, dataset_dict, batch_size, collate_fn, accelerator=None):
        self.model = model
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self._signature_columns = None
        
        for task in dataset_dict:
            dataset_dict[task] = self._remove_unused_columns(dataset_dict[task])
            
        self.dataloader_dict = dict()
        for task in dataset_dict:
            subset = dataset_dict[task]
            self.dataloader_dict[task] = DataLoader(subset, batch_size = self.batch_size, sampler = RandomSampler(subset), collate_fn = self.collate_fn)
        if accelerator is not None:
            for task in self.dataloader_dict:
                self.dataloader_dict[task] = accelerator.prepare(self.dataloader_dict[task])
        
        self.num_batches_dict = {task: len(dataloader) for task, dataloader in self.dataloader_dict.items()}
        self.tasks_list = list(self.dataloader_dict.keys())
        self.num_tasks = len(self.tasks_list)
        
        self.task_max_num_batch = max(list(self.num_batches_dict.values()))
        
        self._init()
        
        
    def _init(self):
        self.dataloader_iters = [iter(dataloader) for task, dataloader in self.dataloader_dict.items()]
        self.batch_idx = 0
            

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + find_labels(self.model.__class__)))
            
    def _remove_unused_columns(self, dataset):
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        return dataset.remove_columns(ignored_columns)
            
    def __len__(self):
        return self.num_tasks * self.task_max_num_batch

    def __iter__(self):
        return self
    
    def __next__(self):
        task_id = self.batch_idx % self.num_tasks
        self.batch_idx += 1
        # if self.batch_idx < self.num_tasks * self.task_max_num_batch:
        try:
            return next(self.dataloader_iters[task_id])
        except StopIteration:
            self.dataloader_iters[task_id] = iter(self.dataloader_dict[self.tasks_list[task_id]])
            return next(self.dataloader_iters[task_id])
        # self._init()

# logger = logging.get_logger(__name__)

class TaskAlternateTrainer(transformers.Seq2SeqTrainer):
    
    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        
        return MultitaskDataloader(self.model, train_dataset, self._train_batch_size, data_collator, self.accelerator)
    