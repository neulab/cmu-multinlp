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
        # collect instances
        insts = [inst for inst in self._dataset_reader._read_one_task(in_path, task=task, e2e=e2e)]

        # predict
        outputs = []
        for batch in tqdm(range(0, len(insts), batch_size)):
            batch = insts[batch:batch + batch_size]
            outputs.extend(self._model.forward_on_instances(batch))

        # output
        incon_count, total_count = 0, 0
        with open(out_path, 'w') as fout:
            for i, output in enumerate(outputs):
                # TODO: better way to handle fake examples?
                if not insts[i]['metadata']['real']:  # skip fake examples
                    continue

                tokens = insts[i]['metadata']['raw_tokens']
                gold_bio = output['gold_bio']
                pred_bio = output['pred_bio']

                if not (len(tokens) == len(gold_bio) and len(tokens) == len(pred_bio)):
                    incon_count += 1
                    logger.debug('length inconsistent tokens: {}, tags: {}'.format(len(tokens), len(gold_bio)))
                total_count += 1

                for t, g, p in zip(tokens, gold_bio, pred_bio):
                    fout.write('{} {} {}\n'.format(t, g, p))
                fout.write('\n')

        logger.warning('{}/{} = {:.3f} samples are inconsistent on the length of tokens and tags'.format(
            incon_count, total_count, incon_count / total_count))
