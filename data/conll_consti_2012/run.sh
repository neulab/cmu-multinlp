#!/usr/bin/env bash
set -e

# prepare conll_dep_2012 first

if [ ! -d ../conll_dep_2012/bracketed ]
then
    echo "'bracketed' does not exist. Preprocess conll_dep_2012 first."
    exit 1
fi

mkdir -p brat/train
mkdir -p brat/dev
mkdir -p brat/test

# implement based on https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/penn_tree_bank.py
python ../bracketed2brat.py --inp ../conll_dep_2012/bracketed/train.txt --out brat/train --num_sent 50
python ../bracketed2brat.py --inp ../conll_dep_2012/bracketed/dev.txt --out brat/dev --num_sent 50
python ../bracketed2brat.py --inp ../conll_dep_2012/bracketed/test.txt --out brat/test --num_sent 50
