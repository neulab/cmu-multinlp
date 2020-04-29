# CMU multinlp

This repository contains scripts to download and preprocess the **G**eneral **L**anguage **A**nalysis **D**atasets (GLAD) benchmark and codes for the task-agnostic SpanRel model, a multi-faceted NLP toolkit that can cover many different tasks.

[Generalizing Natural Language Analysis through Span-relation Representations (ACL2020)](https://arxiv.org/abs/1911.03822)

## Prerequisites

```bash
# install jsonnet from https://github.com/google/jsonnet
conda create -n spanrel python=3.6
conda activate spanrel
./setup.sh
```

## Datasets

8 datasets consisting of annotations of 10 tasks are included in this repository.

| Dataset             | Task    | Task code     | Dir                      |
|---------------------|---------|---------------|--------------------------|
| Wet Lab Protocols   | NER     | wlp           | data/wlp                 |
|                     | RE      | wlp           | data/wlp                 |
| CoNLL-2003          | NER     | ner           | data/semeval_2014/       |
| SemEval-2010 Task 8 | RE      | rc            | data/semeval_2010_task8/ |
| OntoNotes 5.0       | Coref.  | coref         | data/conll_coref_2012/   |
|                     | SRL     | srl           | data/conll_srl_2012/     |
|                     | POS     | pos_conll     | data/conll_pos_2012/     |
|                     | Dep.    | dp_conll      | data/conll_dep_2012/     |
|                     | Consti. | consti_conll  | data/conll_consti_2012/  |
| Penn Treebank       | POS     | pos           | data/ptb_pos/            |
|                     | Dep.    | dp            | data/ptb/                |
|                     | Consti. | consti        | data/ptb_consti/         |
| OIE2016             | OpenIE  | oie           | data/openie/             |
| MPQA 3.0            | ORL     | orl           | data/mpqa/               |
| SemEval-2014 Task 4 | ABSA    | semeval14_st2 | data/semeval_2014/       |

Follow the instructions in `run.sh` in each dataset directory to download and preprocess the datasets into BRAT format.

## Train and Evaluate SpanRel models

Run BERT-based models, where `$emb` can be bert-base-uncased, bert-large-uncased, and `$task` is one of the "task code" shown in the table.
```bash
./run_by_config_bert.sh $task $emb $output
```

Run GloVe/ELMo-based models, where `$emb` can be `glove` or `elmo`, and `$task` is one of the "task code" shown in the table.
```bash
./run_by_config.sh $task $emb $output
```

## Train and Evaluate SpanRel models on other datasets

1. Put the data in `data/kairos` with `train`, `dev`, and `test` sub-directory, each containing multiple `.ann` and `.txt` files.
2. Build vocabulary: `./run_by_config_bert.sh kairos bert-base-uncased output/kairos_vocab`.
3. Modify the `vocab` variable in the `kairos` section of `run_by_config_bert.sh` to `output/kairos_vocab/vocabulary`.
4. Train and evaluate: `./run_by_config_bert.sh kairos bert-base-uncased output/kairos_log`.
