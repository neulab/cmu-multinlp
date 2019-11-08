#!/usr/bin/env bash
set -e

# Access Penn Tree Bank dataset from https://catalog.ldc.upenn.edu/LDC99T42.
# Use https://github.com/hankcs/TreebankPreprocessing to generate tsv files for POS,
# add put them in tsv directory.

if [ ! -d tsv ]
then
    echo "'tsv' does not exist. Download and preprocess PTB first."
    exit 1
fi

mkdir -p brat/train
mkdir -p brat/dev
mkdir -p brat/test

python tsv2brat.py --inp tsv/train.tsv --out brat/train/
python tsv2brat.py --inp tsv/dev.tsv --out brat/dev/
python tsv2brat.py --inp tsv/test.tsv --out brat/test/
