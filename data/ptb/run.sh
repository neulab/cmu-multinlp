#!/usr/bin/env bash
set -e

# Access Penn Tree Bank dataset from https://catalog.ldc.upenn.edu/LDC99T42.
# Use https://github.com/hankcs/TreebankPreprocessing and stanford parser
# (https://nlp.stanford.edu/software/stanford-dependencies.html) to generate CoNLL-X format,
# add put them in conllx directory.
# `java -mx1g edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -keepPunct -conllx -treeFile input > output`
# is faster than the repo.

if [ ! -d conllx ]
then
    echo "'conllx' does not exist. Download and preprocess PTB first."
    exit 1
fi

mkdir -p brat/train
mkdir -p brat/dev
mkdir -p brat/test

# conllXtostandoff.py is adapted from https://github.com/nlplab/brat
python ../conllXtostandoff.py -o brat/train conllx/train.conllx
python ../conllXtostandoff.py -o brat/dev conllx/dev.conllx
python ../conllXtostandoff.py -o brat/test conllx/test.conllx
