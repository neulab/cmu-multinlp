import logging
from typing import List, Tuple, Iterable, Dict, Iterator
from collections import defaultdict

from overrides import overrides

from allennlp.common.util import A
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators.bucket_iterator import sort_by_padding

from .balanced_bucket_iterator import split_by_task, interleave_by_task, BalancedBucketIterator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def ada_task_lazy_groups_of(iterator: Iterator[A],
                            group_size: Dict[str, int],
                            max_total_seq_len: Dict[str, int] = None) -> Iterator[List[A]]:
    if max_total_seq_len is None:
        max_total_seq_len = defaultdict(lambda: None)
    result = []
    max_len = -1
    pre_task = None
    while True:
        try:
            inst = next(iterator)
            cur_task = inst['task_labels'].label
        except StopIteration:
            break
        if max_total_seq_len[cur_task] and len(inst['text']) > max_total_seq_len[cur_task]:
            raise Exception('one sample with length {} is larger than threshold {}'.format(
                len(inst['text']), max_total_seq_len[cur_task]))
        if (pre_task and len(result) >= group_size[pre_task]) or \
                (pre_task and cur_task != pre_task) or \
                (max_total_seq_len[cur_task] and
                 max(max_len, len(inst['text'])) * (len(result) + 1) > max_total_seq_len[cur_task]):
            yield result
            result = []
            max_len = -1
        max_len = max(max_len, len(inst['text']))
        result.append(inst)
        pre_task = cur_task
    if len(result) > 0:
        yield result


@DataIterator.register('ada_balanced_bucket')
class AdaptiveBalancedBucketIterator(BalancedBucketIterator):
    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 task_namespace: str = 'task_labels',
                 # number of samples to draw successively for each task during interleaving
                 num_interleave_per_task: Dict[str, int] = None,
                 max_total_seq_len: Dict[str, int] = None,
                 batch_size_per_task: Dict[str, int] = None,
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
        from different tasks balanced in a batch and adaptively adjust the batch size.
        '''
        super().__init__(sorting_keys,
                         task_namespace=task_namespace,
                         num_interleave_per_task=num_interleave_per_task,
                         padding_noise=padding_noise,
                         biggest_batch_first=biggest_batch_first,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         cache_instances=cache_instances,
                         track_epoch=track_epoch,
                         maximum_samples_per_batch=maximum_samples_per_batch)
        if max_total_seq_len is None:
            max_total_seq_len = defaultdict(lambda: None)
        self._max_total_seq_len = max_total_seq_len
        if batch_size_per_task is None:
            batch_size_per_task = defaultdict(lambda: batch_size)
        self._batch_size_per_task = batch_size_per_task


    @overrides
    def _instance_list_to_batch(self, instances: List[Instance]) -> Iterable[List[Instance]]:
        # split instances by tasks
        inst_li_by_task = split_by_task(instances, task_namespace=self._task_namespace)
        inst_li_by_task = dict((k, sort_by_padding(
            inst_li_by_task[k], self._sorting_keys, self.vocab, self._padding_noise)) for k in inst_li_by_task)

        # interleave instances from different tasks uniformly
        instance_list = interleave_by_task(inst_li_by_task, self._num_interleave_per_task)

        # create batches
        yield from ada_task_lazy_groups_of(iter(instance_list), self._batch_size_per_task, self._max_total_seq_len)
