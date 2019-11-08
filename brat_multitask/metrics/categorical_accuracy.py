from overrides import overrides

from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics import CategoricalAccuracy


@Metric.register('my_categorical_accuracy')
class MyCategoricalAccuracy(CategoricalAccuracy):
    ''' Wrapper of AllenNLP's CategoricalAccuracy with an additional used field '''
    def __init__(self, *args, **kwargs) -> None:
        super(MyCategoricalAccuracy, self).__init__(*args, **kwargs)
        self._used = False


    def __call__(self, *args, **kwargs):
        self._used = True
        super(MyCategoricalAccuracy, self).__call__(*args, **kwargs)


    def get_metric(self, reset: bool = False):
        if not self._used:
            return None
        return super(MyCategoricalAccuracy, self).get_metric(reset)


    @overrides
    def reset(self):
        super(MyCategoricalAccuracy, self).reset()
        self._used = False
