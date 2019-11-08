from typing import List, Dict, Tuple

from overrides import overrides
from collections import defaultdict
import numpy as np
import torch

from allennlp.training.metrics.metric import Metric
from allennlp.data import Vocabulary


@Metric.register('semeval_2010')
class Semeval2010(Metric):
    '''
    compute precision recall and f1 for SemEval 2010 Task 8
    '''
    def __init__(self,
                 vocab: Vocabulary,
                 namespace: str,
                 reduce: str = 'macro',
                 label_cate: List[Tuple] = None) -> None:
        assert reduce in {'micro', 'macro'}
        self._reduce = reduce
        self._vocab = vocab
        self._namespace = namespace
        self._label_cate = label_cate
        if label_cate is None:
            # labels in SemEval 2010 Task 8
            self._label_cate = [
                ('Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)'),
                ('Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)'),
                ('Member-Collection(e1,e2)', 'Member-Collection(e2,e1)'),
                ('Entity-Origin(e1,e2)', 'Entity-Origin(e2,e1)'),
                ('Message-Topic(e1,e2)', 'Message-Topic(e2,e1)'),
                ('Component-Whole(e1,e2)', 'Component-Whole(e2,e1)'),
                ('Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)'),
                ('Content-Container(e1,e2)', 'Content-Container(e2,e1)'),
                ('Product-Producer(e1,e2)', 'Product-Producer(e2,e1)')]

        self._count: Dict[Tuple, Dict] = defaultdict(lambda: {'match': 0, 'predict': 0, 'gold': 0})
        self._used = False


    def __call__(self,
                 predictions: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                 labels: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                 mask: torch.LongTensor  # SHAPE: (batch_size, seq_len)
                 ):
        if len(predictions.size()) != 2:
            raise Exception('inputs should have two dimensions')
        self._used = True

        # compute three values for each category
        for cate in self._label_cate:
            for label in cate:
                label = self._vocab.get_token_index(label, self._namespace)
                self._count[cate]['gold'] += \
                    (mask * labels.eq(label).long()).sum().item()
                self._count[cate]['predict'] += \
                    (mask * predictions.eq(label).long()).sum().item()
                self._count[cate]['match'] += \
                    (mask * labels.eq(label).long() * labels.eq(predictions).long()).sum().item()


    def get_metric(self, reset: bool = False):
        if not self._used:  # return None when the metric has not been called
            return None
        if self._reduce == 'micro':
            match, predict, gold = 0, 0, 0
            for cate, count in self._count:
                match += count['match']
                predict += count['predict']
                gold += count['gold']
            p = match / (predict + 1e-10)
            r = match / (gold + 1e-10)
            f = 2 * p * r / (p + r + 1e-10)
        elif self._reduce == 'macro':
            ps, rs, fs = [], [], []
            for cate, count in self._count.items():
                p = count['match'] / (count['predict'] + 1e-10)
                r = count['match'] / (count['gold'] + 1e-10)
                f = 2 * p * r / (p + r + 1e-10)
                ps.append(p)
                rs.append(r)
                fs.append(f)
            p = np.mean(ps)
            r = np.mean(rs)
            f = np.mean(fs)
        else:
            raise NotImplementedError
        if reset:
            self.reset()
        return {'p': p, 'r': r, 'f': f}


    @overrides
    def reset(self):
        self._count = defaultdict(lambda: {'match': 0, 'predict': 0, 'gold': 0})
        self._used = False
