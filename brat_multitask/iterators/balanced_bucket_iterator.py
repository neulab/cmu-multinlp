import logging
from typing import List, Tuple, Iterable, Dict
from collections import defaultdict
from overrides import overrides

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators.bucket_iterator import sort_by_padding

from .filter_bucket_iterator import FilterBucketIterator

import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def split_by_task(instance_li: List[Instance], task_namespace: str) -> Dict[str, List[Instance]]:
    result = {}
    for inst in instance_li:
        task = inst[task_namespace].label
        if task not in result:
            result[task] = []
        result[task].append(inst)
    return result


def interleave_by_task(inst_li_by_task: Dict[str, List[Instance]], num_per_task: Dict[str, int]):
    task_len_li = [(k, len(inst_li_by_task[k])) for k in inst_li_by_task]
    num_inst = np.sum([l[1] for l in task_len_li])
    ideal_dist = np.array([l[1] / num_inst for l in task_len_li])
    task_ind = np.zeros_like(ideal_dist, dtype=int)
    result = []
    total = 0
    while total < num_inst:
        task = np.argmax(ideal_dist - task_ind / (np.sum(task_ind) + 1e-5))
        task_name = task_len_li[task][0]
        li = inst_li_by_task[task_name][task_ind[task]:task_ind[task] + num_per_task[task_name]]
        result.extend(li)
        task_ind[task] += len(li)
        total += len(li)
    for i, tl in enumerate(task_ind):
        assert len(inst_li_by_task[task_len_li[i][0]]) == tl, 'task interleave failure'
    return result


@DataIterator.register('balanced_bucket')
class BalancedBucketIterator(FilterBucketIterator):
    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 task_namespace: str = 'task_labels',
                 # number of samples to draw successively for each task during interleaving
                 num_interleave_per_task: Dict[str, int] = None,
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None) -> None:
        '''
        Use with multitask learning to make the number of samples
        from different tasks balanced in a batch.
        '''
        super().__init__(sorting_keys,
                         padding_noise=padding_noise,
                         biggest_batch_first=biggest_batch_first,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         cache_instances=cache_instances,
                         track_epoch=track_epoch,
                         maximum_samples_per_batch=maximum_samples_per_batch)
        self._task_namespace = task_namespace
        if num_interleave_per_task is None:
            num_interleave_per_task = defaultdict(lambda: 1)
        self._num_interleave_per_task = num_interleave_per_task


    @overrides
    def _instance_list_to_batch(self, instances: List[Instance]) -> Iterable[List[Instance]]:

        # split instances by tasks
        inst_li_by_task = split_by_task(instances, task_namespace=self._task_namespace)
        inst_li_by_task = dict((k, sort_by_padding(
            inst_li_by_task[k], self._sorting_keys, self.vocab, self._padding_noise)) for k in inst_li_by_task)

        # interleave instances from different tasks uniformly
        instance_list = interleave_by_task(inst_li_by_task, self._num_interleave_per_task)

        # create batches
        yield from lazy_groups_of(iter(instance_list), self._batch_size)
