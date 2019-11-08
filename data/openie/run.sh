#!/usr/bin/env bash
set -e

# download OIE2016 data
git clone https://github.com/jzbjyb/oie_rank.git

mkdir -p brat/train
mkdir -p brat/dev
mkdir -p brat/test

# convert files from conll 2012 format to brat format
python conll2brat.py --inp oie_rank/data/train --out brat/train --merge
python conll2brat.py --inp oie_rank/data/dev --out brat/dev --merge
python conll2brat.py --inp oie_rank/data/test --out brat/test --merge
