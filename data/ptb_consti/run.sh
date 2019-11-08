#!/usr/bin/env bash
set -e

# Access Penn Tree Bank dataset from https://catalog.ldc.upenn.edu/LDC99T42.
# Use https://github.com/hankcs/TreebankPreprocessing to generate bracketed files,
# add put them in bracketed directory.

if [ ! -d bracketed ]
then
    echo "'bracketed' does not exist. Download and preprocess PTB first."
    exit 1
fi

mkdir -p brat/train
mkdir -p brat/dev
mkdir -p brat/test

# implement based on https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/penn_tree_bank.py
python ../bracketed2brat.py --inp bracketed/train.txt --out brat/train
python ../bracketed2brat.py --inp bracketed/dev.txt --out brat/dev
python ../bracketed2brat.py --inp bracketed/test.txt --out brat/test
