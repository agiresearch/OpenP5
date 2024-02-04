from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
from processor.Collator import Collator, calculate_whole_word_ids
import math


class SingleMultiDataTaskSampler(Sampler):
    def parse_sampler_args(parser):
        """
        parse sampler related command line arguments
        """
        parser.add_argument("--batch_size", type=int, default=32, help="batch size")
        parser.add_argument("--eval_batch_size", type=int, default=32, help="the batch size for evaluation")
        parser.add_argument("--dist_sampler", type=int, default=0, help='use DistributedSampler if 1, otherwise use our own sampler.')
        
        return parser
    
    def __init__(self, dataset, batch_size, seed, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.dataset_task_size = []
        for ds in self.dataset.datasets:
            for task in ds.task_data:
                self.dataset_task_size.append(len(ds.task_data[task]))
        self.largest_task_size = max(self.dataset_task_size)
    def __iter__(self):
        data_list = []
        iterator_list = []
        for i in range(len(self.dataset.datasets)):
            ds = self.dataset.datasets[i]
            if self.shuffle:
                ds.shuffle(self.seed + self.epoch)
            for task in ds.task_data:
                # data_ = SequentialSampler(ds.task_data[task])
                data_list.append(ds.task_data[task])
                # iterator = sampler.__iter__()
                iterator = iter(ds.task_data[task])
                iterator_list.append(iterator)
        
        cum_index = [0] + self.dataset.cumulative_sizes[:-1]
        task_cum_index = []
        for i in range(len(self.dataset.datasets)):
            ds = self.dataset.datasets[i]
            cur_cum_index = cum_index[i]
            for task in ds.task_data:
                task_cum_index.append(cur_cum_index)
                # cur_cum_index += len(ds.task_data[task])
                
                
        step = self.batch_size * len(self.dataset_task_size)
        epoch_data_size = self.largest_task_size * len(self.dataset_task_size)
        
        final_list = []
        for _ in range(0, epoch_data_size, step):
            for i in range(len(self.dataset_task_size)):
                cur_iterator = iterator_list[i]
                cur_samples = []
                for _ in range(self.batch_size):
                    try:
                        cur_element = cur_iterator.__next__()
                        cur_element += task_cum_index[i]
                        cur_samples.append(cur_element)
                        
                    except StopIteration:
                        iterator_list[i] = iter(data_list[i])
                        cur_iterator = iterator_list[i]
                        cur_element = cur_iterator.__next__()
                        cur_element += task_cum_index[i]
                        cur_samples.append(cur_element)
                final_list.extend(cur_samples)
        
        # print(len(set(final_list)))
        return iter(final_list)
    
    def __len__(self):
        return self.batch_size * math.ceil(self.largest_task_size / self.batch_size) * len(self.dataset_task_size)
    
    def set_epoch(self, epoch):
        self.epoch = epoch