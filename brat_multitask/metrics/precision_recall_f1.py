from typing import List, Union

from overrides import overrides
import torch
import numpy as np
from collections import defaultdict

from allennlp.training.metrics.metric import Metric


@Metric.register('precision_recall_f1')
class PrecisionRecallF1(Metric):
    '''
    compute precision recall and f1
    '''
    def __init__(self, neg_label: int, reduce: str = 'micro', binary_match: bool = False) -> None:
        self._neg_label = neg_label
        assert reduce in {'micro', 'macro'}
        if reduce == 'macro':
            raise Exception('precision, recall, and F1 don\'t have macro version?')
        self._reduce = reduce
        self._binary_match = binary_match  # only consider the existence
        self._count = 0
        self._precision = 0
        self._recall = 0
        self._bucket = {
            'count': defaultdict(lambda: 0),
            'precision': defaultdict(lambda: 0),
            'recall': defaultdict(lambda: 0)
        }
        self._recall_local = 0
        self._f1 = 0
        self._num_sample = 0
        self._used = False


    def __call__(self,
                 predictions: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                 labels: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                 mask: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                 recall: torch.LongTensor = None,  # SHAPE: (batch_size, seq_len)
                 duplicate_check: bool = True,
                 bucket_value: torch.LongTensor = None,  # SHPAE: (batch_size, seq_len)
                 ):  # SHAPE: (batch_size)
        if len(predictions.size()) != 2:
            raise Exception('inputs should have two dimensions')
        self._used = True
        predicted = (predictions.ne(self._neg_label).long() * mask).float()
        whole_subset = (labels.ne(self._neg_label).long() * mask).float()
        self._recall_local += whole_subset.sum().item()
        if recall is not None:
            whole = recall.float()
            if duplicate_check:
                assert whole_subset.sum().item() <= whole.sum().item(), 'found duplicate span pairs'
        else:
            whole = whole_subset

        if self._binary_match:
            matched = (predictions.ne(self._neg_label).long() * labels.ne(self._neg_label).long() * mask).float()
        else:
            matched = (predictions.eq(labels).long() * mask * predictions.ne(self._neg_label).long()).float()

        if self._reduce == 'micro':
            self._count += matched.sum().item()
            self._precision += predicted.sum().item()
            self._recall += whole.sum().item()
            self._num_sample += predictions.size(0)
            if bucket_value is not None:
                bucket_value = (bucket_value * mask).cpu().numpy().reshape(-1)
                matched = matched.cpu().numpy().reshape(-1)
                predicted = predicted.cpu().numpy().reshape(-1)
                whole_subset = whole_subset.cpu().numpy().reshape(-1)
                count = np.bincount(bucket_value, matched)
                precision = np.bincount(bucket_value, predicted)
                recall = np.bincount(bucket_value, whole_subset)
                for name in ['count', 'precision', 'recall']:
                    value = eval(name)
                    for b, v in zip(range(len(value)), value):
                        self._bucket[name][b] += v

        elif self._reduce == 'macro':
            # TODO: the implementation is problematic because samples without label/prediction
            #   are counted as zero recall/precision
            self._count += matched.size(0)
            pre = matched / (predicted + 1e-10)
            rec = matched / (whole + 1e-10)
            f1 = 2 * pre * rec / (pre + rec + 1e-10)
            self._precision += pre.sum().item()
            self._recall += rec.sum().item()
            self._f1 += f1.sum().item()
            self._num_sample += predictions.size(0)


    def get_metric(self, reset: bool = False):
        if not self._used:  # return None when the metric has not been called
            return None
        if self._reduce == 'micro':
            p = self._count / (self._precision + 1e-10)
            r = self._count / (self._recall + 1e-10)
            f = 2 * p * r / (p + r + 1e-10)
            included_recall = self._recall_local / (self._recall + 1e-10)
            m = {'p': p, 'r': r, 'f': f, 'r_': included_recall}
            if len(self._bucket['count']) > 0:
                for b, v in self._bucket['count'].items():
                    p = self._bucket['count'][b] / (self._bucket['precision'][b] + 1e-10)
                    r = self._bucket['count'][b] / (self._bucket['recall'][b] + 1e-10)
                    f = 2 * p * r / (p + r + 1e-10)
                    m['bucket_f_{}'.format(b)] = f
                for b, v in self._bucket['count'].items():
                    m['bucket_all_{}'.format(b)] = self._bucket['recall'][b]
        elif self._reduce == 'macro':
            m = {
                'p': self._precision / (self._count + 1e-10),
                'r': self._recall / (self._count + 1e-10),
                'f': self._f1 / (self._count + 1e-10),
            }
        if reset:
            self.reset()
        return m


    @overrides
    def reset(self):
        self._count = 0
        self._precision = 0
        self._recall = 0
        self._bucket = {
            'count': defaultdict(lambda: 0),
            'precision': defaultdict(lambda: 0),
            'recall': defaultdict(lambda: 0)
        }
        self._recall_local = 0
        self._f1 = 0
        self._num_sample = 0
        self._used = False
