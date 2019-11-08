from typing import List
from overrides import overrides
from allennlp.training.metrics.metric import Metric


@Metric.register('multiple_loss')
class MultipleLoss(Metric):
    '''
    Maintain multiple loss values
    '''
    def __init__(self, loss_names: List[str]) -> None:
        self._loss_names = loss_names
        self._losses = dict((n, 0.0) for n in self._loss_names)
        self._count = dict((n, 0) for n in self._loss_names)


    def __call__(self, name: str, loss: float, count: int):
        self._losses[name] += loss
        self._count[name] += count


    def get_metric(self, reset: bool = False):
        losses = dict((n + '_loss', self._losses[n] / (self._count[n] + 1e-10)) for n in self._loss_names)
        if reset:
            self.reset()
        return losses


    @overrides
    def reset(self):
        self._losses = dict((n, 0.0) for n in self._loss_names)
        self._count = dict((n, 0) for n in self._loss_names)
