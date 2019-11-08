import logging
from typing import List, Tuple, Iterable, Iterator

from overrides import overrides

from allennlp.common.util import A
from allennlp.data.instance import Instance
from allennlp.data.iterators.bucket_iterator import sort_by_padding
from allennlp.data.iterators.data_iterator import DataIterator

from brat_multitask.iterators import FilterBucketIterator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def ada_lazy_groups_of(iterator: Iterator[A], group_size: int, max_total_seq_len: int = None) -> Iterator[List[A]]:
    result = []
    max_len = -1
    while True:
        try:
            inst = next(iterator)
        except StopIteration:
            break
        if max_total_seq_len and len(inst['text']) > max_total_seq_len:
            raise Exception('one sample with length {} is larger than threshold {}'.format(
                len(inst['text']), max_total_seq_len))
        if len(result) >= group_size or \
                (max_total_seq_len and max(max_len, len(inst['text'])) * (len(result) + 1) > max_total_seq_len):
            yield result
            result = []
            max_len = -1
        max_len = max(max_len, len(inst['text']))
        result.append(inst)
    if len(result) > 0:
        yield result


@DataIterator.register('ada_bucket')
class AdaptiveBucketIterator(FilterBucketIterator):
    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 max_total_seq_len: int = None,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None) -> None:
        super().__init__(sorting_keys,
                         padding_noise=padding_noise,
                         biggest_batch_first=biggest_batch_first,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         cache_instances=cache_instances,
                         track_epoch=track_epoch,
                         maximum_samples_per_batch=maximum_samples_per_batch)
        self._max_total_seq_len = max_total_seq_len


    @overrides
    def _instance_list_to_batch(self, instances: List[Instance]) -> Iterable[List[Instance]]:
        instances = sort_by_padding(instances, self._sorting_keys, self.vocab, self._padding_noise)
        yield from ada_lazy_groups_of(iter(instances), self._batch_size, self._max_total_seq_len)
