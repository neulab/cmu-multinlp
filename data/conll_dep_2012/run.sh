#!/usr/bin/env bash
set -e

# Set ONTONOTES_ROOT to the root of the ontonotes dataset, e.g., "ontonotes5/conll-formatted-ontonotes-5.0/data"
# Follow the instructions on http://cemantix.org/data/ontonotes.html to prepare ontonotes dataset.

# Filter ontonotes for end2end experiments (adapted from https://github.com/luheng/lsgn/)

mkdir -p conll12_ids

wget http://conll.cemantix.org/2012/download/ids/english/coref/train.id -O conll12_ids/train
wget http://conll.cemantix.org/2012/download/ids/english/coref/development.id -O conll12_ids/dev
wget http://conll.cemantix.org/2012/download/ids/english/coref/test.id -O conll12_ids/test

# generate tree format

mkdir -p bracketed

python conll2tree.py --inp ${ONTONOTES_ROOT}/train/data/ --out bracketed/train.txt --filter conll12_ids/train
python conll2tree.py --inp ${ONTONOTES_ROOT}/development/data/ --out bracketed/dev.txt --filter conll12_ids/dev
python conll2tree.py --inp ${ONTONOTES_ROOT}/test/data/ --out bracketed/test.txt --filter conll12_ids/test

# conver to dependency
# set STANFORD_PARSER to the jar, e.g., "stanford-parser-full-2013-11-12/stanford-parser.jar"
mkdir conllx

java -cp ${STANFORD_PARSER} -mx1g edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -keepPunct -conllx -treeFile bracketed/train.txt > conllx/train.conllx
java -cp ${STANFORD_PARSER} -mx1g edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -keepPunct -conllx -treeFile bracketed/dev.txt > conllx/dev.conllx
java -cp ${STANFORD_PARSER} -mx1g edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -keepPunct -conllx -treeFile bracketed/test.txt > conllx/test.conllx

# convert to brat
mkdir -p brat/train
mkdir -p brat/dev
mkdir -p brat/test

# conllXtostandoff.py is adapted from https://github.com/nlplab/brat
python ../conllXtostandoff.py -o brat/train conllx/train.conllx sent:50
python ../conllXtostandoff.py -o brat/dev conllx/dev.conllx sent:50
python ../conllXtostandoff.py -o brat/test conllx/test.conllx sent:50
