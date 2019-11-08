#!/usr/bin/env bash
set -e

# download data from google drive
gdown.pl "https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?usp=sharing" SemEval2010_task8_all_data.zip
unzip SemEval2010_task8_all_data.zip

mkdir -p brat/train
mkdir -p brat/dev
mkdir -p brat/test

python raw2brat.py --inp SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT --out brat/train/
python raw2brat.py --inp SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT --out brat/test
python ../split.py --inp brat/train/ --out brat/dev/ --ratio 0.2
