#!/usr/bin/env bash
#SBATCH --mem=30000
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out

emb=$2
output=$3
args="${@:4}"

# default configs
use_neg=true
max_sent_len=null
tokenizer=space
sentencizer=newline
truncate_span_loss=true
pair_ind_method=null
special_loss=false
use_attentive_span_repr=true
use_context_layer=true
batch_size=32
max_total_seq_len=1600
bucket_widths=false
max_num_sample=null
num_order=0

if [[ $emb == 'glove' ]]; then
    attentive_after_context=true
elif [[ $emb == 'elmo' ]]; then
    attentive_after_context=false
fi

if [[ $1 == 'dp' ]]; then
    task=dp
    max_span_width=1
    data_dir=ptb/brat
    task_loss=span_pair
    spans_per_word=1.0
    validation_metric=dp_sp_prf_f

elif [[ $1 == 'dp_conll' ]]; then
    task=dp_conll
    max_span_width=1
    data_dir=conll_dep_2012/brat
    task_loss=span_pair
    spans_per_word=1.0
    validation_metric=dp_conll_sp_prf_f

elif [[ $1 == 'srl' ]]; then
    task=srl
    max_span_width=30
    data_dir=conll_srl_2012/brat
    task_loss=span-span_pair
    spans_per_word=1.0
    max_total_seq_len=512
    validation_metric=srl_sp_prf_f

elif [[ $1 == 'ner' ]]; then
    task=ner
    max_span_width=10
    tokenizer=spacy
    data_dir=conll_ner_2003/brat
    task_loss=span
    spans_per_word=0.4
    validation_metric=ner_s_prf_f

elif [[ $1 == 'coref' ]]; then
    task=coref
    max_span_width=10
    sentencizer=concat
    data_dir=conll_coref_2012/brat
    task_loss=span-span_pair
    spans_per_word=0.4
    pair_ind_method=left:100
    special_loss=true
    max_total_seq_len=4096
    validation_metric=coref_coref_2

elif [[ $1 == 'consti' ]]; then
    task=consti
    max_span_width=null
    data_dir=ptb_consti/brat
    task_loss=span
    spans_per_word=1000000.0
    use_attentive_span_repr=false
    validation_metric=consti_bracket_evalb_f1_measure
    bucket_widths=true

elif [[ $1 == 'consti_conll' ]]; then
    task=consti_conll
    max_span_width=null
    data_dir=conll_consti_2012/brat
    task_loss=span
    spans_per_word=1000000.0
    use_attentive_span_repr=false
    validation_metric=consti_conll_bracket_evalb_f1_measure
    bucket_widths=true

elif [[ $1 == 'consti_small' ]]; then
    task=consti
    max_span_width=10
    data_dir=ptb_consti/brat
    task_loss=span
    spans_per_word=2.0
    validation_metric=consti_bracket_evalb_f1_measure

elif [[ $1 == 'consti_conll_small' ]]; then
    task=consti_conll
    max_span_width=10
    data_dir=conll_consti_2012/brat
    task_loss=span
    spans_per_word=2.0
    validation_metric=consti_conll_bracket_evalb_f1_measure

elif [[ $1 == 'rc' ]]; then
    task=rc
    max_span_width=5
    tokenizer=spacy
    data_dir=semeval_2010_task8/brat
    task_loss=span-span_pair
    spans_per_word=5
    validation_metric=rc_sp_prf_r

elif [[ $1 == 'oie' ]]; then
    task=oie
    max_span_width=30
    data_dir=openie/brat
    task_loss=span-span_pair
    spans_per_word=0.8
    max_total_seq_len=512
    validation_metric=oie_sp_prf_f

elif [[ $1 == 'pos' ]]; then
    task=pos
    use_neg=false
    max_span_width=1
    data_dir=ptb_pos/brat
    task_loss=span
    truncate_span_loss=false
    spans_per_word=0
    max_total_seq_len=1000000
    validation_metric=pos_s_acc

elif [[ $1 == 'pos_conll' ]]; then
    task=pos_conll
    use_neg=false
    max_span_width=1
    data_dir=conll_pos_2012/brat
    task_loss=span
    truncate_span_loss=false
    spans_per_word=0
    max_total_seq_len=1000000
    validation_metric=pos_conll_s_acc

elif [[ $1 == 'wlp' ]]; then
    task=wlp
    max_span_width=10
    tokenizer=spacy
    data_dir=wlp/WLP-Dataset
    task_loss=span-span_pair
    spans_per_word=0.6
    validation_metric=wlp_sp_prf_f

elif [[ $1 == 'semeval14_st2' ]]; then
    task=semeval14_st2
    max_span_width=10
    tokenizer=spacy
    data_dir=semeval_2014/brat
    task_loss=span-span_pair
    spans_per_word=0.4
    validation_metric=semeval14_st2_s_prf_f

elif [[ $1 == 'orl' ]]; then
    task=orl
    max_span_width=30
    data_dir=mpqa/brat
    task_loss=span-span_pair
    spans_per_word=0.3
    max_total_seq_len=512
    validation_metric=orl_sp_prf_f

elif [[ $1 == 'orl_gold' ]]; then
    task=orl
    max_span_width=30
    data_dir=mpqa/brat
    task_loss=span-span_pair
    pair_ind_method=gold_predicate
    spans_per_word=1.0
    max_total_seq_len=512
    validation_metric=orl_sp_prf_f

fi

temp_file=$(mktemp)

# build json
jsonnet \
    --ext-str task=${task} \
    --ext-code use_neg=${use_neg} \
    --ext-code max_span_width=${max_span_width} \
    --ext-code max_sent_len=${max_sent_len} \
    --ext-str tokenizer=${tokenizer} \
    --ext-str sentencizer=${sentencizer} \
    --ext-str data_dir=${data_dir} \
    --ext-str task_loss=${task_loss} \
    --ext-code truncate_span_loss=${truncate_span_loss} \
    --ext-str spans_per_word=${spans_per_word} \
    --ext-str pair_ind_method=${pair_ind_method} \
    --ext-code special_loss=${special_loss} \
    --ext-code use_attentive_span_repr=${use_attentive_span_repr} \
    --ext-code batch_size=${batch_size} \
    --ext-code max_total_seq_len=${max_total_seq_len} \
    --ext-str validation_metric=${validation_metric} \
    --ext-code bucket_widths=${bucket_widths} \
    --ext-code attentive_after_context=${attentive_after_context} \
    --ext-code use_context_layer=${use_context_layer} \
    --ext-code max_num_sample=${max_num_sample} \
    --ext-code num_order=${num_order} \
    training_config/template/${emb}.jsonnet > ${temp_file}

echo "write config to" ${temp_file}
cat ${temp_file}

allennlp train ${temp_file} -s ${output} --include-package brat_multitask ${args}
