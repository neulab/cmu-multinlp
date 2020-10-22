import sys
import os
import argparse

from allennlp.common.util import import_submodules
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict using brat model')
    parser.add_argument('--model', type=str, help='model file', required=True)
    parser.add_argument('--inp', type=str, help='input file', required=True)
    parser.add_argument('--out', type=str, help='output file.', required=True)
    parser.add_argument('--task', type=str, help='task', required=True)
    parser.add_argument('--cuda_device', type=int, default=0, help='id of GPU to use (if any)')
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(__file__, '../..'))
    sys.path.insert(0, root_dir)

    import_submodules('brat_multitask')

    arc = load_archive(args.model, cuda_device=args.cuda_device)
    predictor = Predictor.from_archive(arc, predictor_name='brat')
    predictor.predict_from_file(args.inp, args.out, task=args.task, e2e=True, batch_size=8)
