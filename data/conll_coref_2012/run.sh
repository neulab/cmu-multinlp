#!/usr/bin/env bash
set -e

# prepare dataset using https://github.com/allenai/allennlp/blob/master/scripts/compile_coref_data.sh
# put the resulting train/dev/test file in conll dir

mkdir -p brat/train
mkdir -p brat/dev
mkdir -p brat/test

python conll2brat.py --inp conll/train.english.v4_gold_conll --out brat/train
python conll2brat.py --inp conll/dev.english.v4_gold_conll --out brat/dev
python conll2brat.py --inp conll/test.english.v4_gold_conll --out brat/test
