from tqdm import tqdm
import logging

from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

logger = logging.getLogger(__name__)


@Predictor.register('brat')
class BratPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)


    def predict_from_file(self,
                          in_path: str,
                          out_path: str,
                          task: str = None,
                          e2e: bool = True,
                          batch_size: int = 256):

        def batch_iter():
            batch = []
            for inst in self._dataset_reader._read_one_task(in_path, task=task, e2e=e2e):
                batch.append(inst)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0:
                yield batch

        print('predict ...')
        incon_count, total_count = 0, 0
        with open(out_path, 'w') as fout:
            for batch in batch_iter():
                for i, output in enumerate(self._model.forward_on_instances(batch)):
                    # TODO: better way to handle fake examples?
                    if not batch[i]['metadata']['real']:  # skip fake examples
                        continue

                    tokens = batch[i]['metadata']['raw_tokens']
                    spans = output['span_with_label']
                    if 'span_pair_with_label' not in output:
                        span_pairs = []
                    else:
                        span_pairs = output['span_pair_with_label']

                    fout.write('{}\t\t{}\t\t{}\n'.format(
                        ' '.join(tokens),
                        '\t'.join(map(lambda span: '{},{},{}'.format(
                            '{},{}'.format(*span[0]), span[1], span[2]), spans)),
                        '\t'.join(map(lambda sp: '{},{},{},{}'.format(
                            '{},{}'.format(*sp[0]), '{},{}'.format(*sp[1]), sp[2], sp[3]), span_pairs)),
                    ))

                    '''
                    # bio output
                    gold_bio = output['gold_bio']
                    pred_bio = output['pred_bio']

                    if not (len(tokens) == len(gold_bio) and len(tokens) == len(pred_bio)):
                        incon_count += 1
                        logger.debug('length inconsistent tokens: {}, tags: {}'.format(len(tokens), len(gold_bio)))
                    total_count += 1

                    for t, g, p in zip(tokens, gold_bio, pred_bio):
                        fout.write('{} {} {}\n'.format(t, g, p))
                    fout.write('\n')
                    '''

        logger.warning('{}/{} = {:.3f} samples are inconsistent on the length of tokens and tags'.format(
            incon_count, total_count, incon_count / (total_count or 1)))
