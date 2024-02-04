from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
from processor.Collator import Collator, calculate_whole_word_ids
from torch.utils.data.distributed import DistributedSampler
from processor.SingleMultiDataTaskSampler import SingleMultiDataTaskSampler
import math


class DistMultiDataTaskSampler(SingleMultiDataTaskSampler):
    
    def __init__(self, dataset, batch_size, num_replicas, rank, seed, shuffle=True):
        super().__init__(dataset, batch_size, seed)
        self.num_replicas = num_replicas
        self.rank = rank
        self.dataset_task_size = []
        for ds in self.dataset.datasets:
            for task in ds.task_data:
                self.dataset_task_size.append(math.ceil(len(ds.task_data[task]) / self.num_replicas))
        self.largest_task_size = max(self.dataset_task_size)
        
        self.shuffle = shuffle
        
    def __iter__(self):
        data_list = []
        iterator_list = []
        for i in range(len(self.dataset.datasets)):
            ds = self.dataset.datasets[i]
            if self.shuffle:
                ds.shuffle(self.seed + self.epoch)
            for task in ds.task_data:
                data_list.append(ds.task_data[task][self.rank::self.num_replicas])
                # sampler = SequentialSampler(ds.task_data[task][self.rank::self.num_replicas])
                # sampler_list.append(sampler)
                iterator = iter(ds.task_data[task][self.rank::self.num_replicas])
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
        # print('rank is {}: '.format(self.rank) + ','.join([str(i) for i in final_list[:20]]))
        return iter(final_list)
    
    def __len__(self):
        return self.batch_size * math.ceil(self.largest_task_size / self.batch_size) * len(self.dataset_task_size)
    