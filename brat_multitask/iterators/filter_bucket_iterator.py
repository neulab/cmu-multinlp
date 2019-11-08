from typing import List, Tuple, Iterable, Deque
import logging
from overrides import overrides
from collections import deque
import random

from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators import BucketIterator

logger = logging.getLogger(__name__)


class FilterBucketIterator(BucketIterator):
    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
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


    @staticmethod
    def is_instance_real(instance: Instance):
        if 'metadata' in instance and 'real' in instance['metadata'] and not instance['metadata']['real']:
            return False
        return True


    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        for instance_list in self._memory_sized_lists(instances):

            # filter fake instances
            instance_list = [inst for inst in instance_list if self.is_instance_real(inst)]

            batches = []
            excess: Deque[Instance] = deque()
            for batch_instances in self._instance_list_to_batch(instance_list):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    batches.append(Batch(possibly_smaller_batches))
            if excess:
                batches.append(Batch(excess))

            # TODO(brendanr): Add multi-GPU friendly grouping, i.e. group
            # num_gpu batches together, shuffle and then expand the groups.
            # This guards against imbalanced batches across GPUs.
            move_to_front = self._biggest_batch_first and len(batches) > 1
            if move_to_front:
                # We'll actually pop the last _two_ batches, because the last one might not be full.
                last_batch = batches.pop()
                penultimate_batch = batches.pop()
            if shuffle:
                # NOTE: if shuffle is false, the data will still be in a different order
                # because of the bucket sorting.
                random.shuffle(batches)
            if move_to_front:
                batches.insert(0, penultimate_batch)
                batches.insert(0, last_batch)

            logger.info('create {} batches from {} instances'.format(len(batches), len(instance_list)))

            yield from batches


    def _instance_list_to_batch(self, instances: List[Instance]) -> Iterable[List[Instance]]:
        raise NotImplemented
