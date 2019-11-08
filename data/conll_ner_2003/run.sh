#!/usr/bin/env bash
set -e

git clone https://github.com/glample/tagger.git
mv tagger/dataset preprocessed

mkdir -p brat/train
mkdir -p brat/dev
mkdir -p brat/test

# conll02tostandoff.py is adapted from https://github.com/nlplab/brat
python conll02tostandoff.py -o brat/train preprocessed/eng.train
python conll02tostandoff.py -o brat/dev preprocessed/eng.testa
python conll02tostandoff.py -o brat/test preprocessed/eng.testb
