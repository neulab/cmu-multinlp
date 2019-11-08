#!/usr/bin/env bash
set -e

# Set ONTONOTES_ROOT to the root of the ontonotes dataset, e.g., "ontonotes5/conll-formatted-ontonotes-5.0/data"
# Follow the instructions on http://cemantix.org/data/ontonotes.html to prepare ontonotes dataset.

# Filter ontonotes for end2end experiments (adapted from https://github.com/luheng/lsgn/)

mkdir -p conll12_ids

wget http://conll.cemantix.org/2012/download/ids/english/coref/train.id -O conll12_ids/train
wget http://conll.cemantix.org/2012/download/ids/english/coref/development.id -O conll12_ids/dev
wget http://conll.cemantix.org/2012/download/ids/english/coref/test.id -O conll12_ids/test

mkdir -p brat/train
mkdir -p brat/dev
mkdir -p brat/test

python conll2brat.py --inp ${ONTONOTES_ROOT}/train/data/ --out brat/train/ --filter conll12_ids/train
python conll2brat.py --inp ${ONTONOTES_ROOT}/development/data/ --out brat/dev/ --filter conll12_ids/dev
python conll2brat.py --inp ${ONTONOTES_ROOT}/test/data/ --out brat/test/ --filter conll12_ids/test
