#!/usr/bin/env python

from typing import Dict, List, Tuple
import argparse
import os
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from nltk.tree import Tree
from tqdm import tqdm
import pickle
from collections import defaultdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser('convert pkl format into brat format')
    parser.add_argument('--inp', type=str, required=True, help='input file')
    parser.add_argument('--out', type=str, required=True, help='output file')
    args = parser.parse_args()

    num_sent_per_doc = 50
    num_bug_sent = 0
    num_holder = 0
    num_target = 0

    data_file, vocab_file = args.inp.split(':')
    with open(data_file, 'rb') as fin:
        data = pickle.load(fin)
    with open(vocab_file, 'rb') as fin:
        vocab = pickle.load(fin)
    win2word = dict((v, k) for k, v in vocab.items())
    print('vocab size {}'.format(len(vocab)))

    # order sentence by str
    data = sorted(data, key=lambda s: ' '.join(win2word[t].replace(' ', '_') for t in s[0]))

    doc_id, sent_count, sent_offset, spanid, span_pair_id = 1, 0, 0, 1, 1
    txt_file = ann_file = None

    ds_li, h_li_li, t_li_li = [], [], []
    sent_seen = set()
    num_dup_pred = 0
    sent2pred: Dict[str, set] = defaultdict(set)
    prev_sent = None
    data.append([None] * 7)  # save last sentence

    for sent, labels, ds_ind, ds_len, ctx, ctx_len, mask in data:
        tokens = [win2word[t].replace(' ', '_') for t in sent] if sent else ['']
        sent_str = ' '.join(tokens)

        # TODO debug
        #if sent_str != '' and not sent_str.startswith("although developed countries are the world 's main source"):
        #    continue

        if sent_str in sent_seen and sent_str != prev_sent:
            raise Exception('the same sentence not adjacent to each other')

        # save one unique sentence
        if prev_sent is not None and sent_str != prev_sent:
            if len(ds_li) == 0:
                raise Exception('empty sentence without predicate')

            num_holder += sum([len(h_li) for h_li in h_li_li])
            num_target += sum([len(t_li) for t_li in t_li_li])

            # token ind to end char ind (exclusive)
            tind2cind: Dict[int, int] = {}
            prev_tokens = prev_sent.split(' ')
            for i, tok in enumerate(prev_tokens):
                if i == 0:
                    tind2cind[i] = len(tok)
                else:
                    tind2cind[i] = tind2cind[i - 1] + len(tok) + 1  # space

            if txt_file is None:
                txt_file = open(os.path.join(args.out, '{}.txt'.format(doc_id)), 'w')
                ann_file = open(os.path.join(args.out, '{}.ann'.format(doc_id)), 'w')

            txt_file.write('{}\n'.format(prev_sent))

            span_startend2tid: Dict[Tuple[int, int], str] = {}
            for ds, h_li, t_li in zip(ds_li, h_li_li, t_li_li):
                tid_label: List[Tuple[int, str]] = []
                for span, label in [(ds, 'sentiment')] + list(zip(h_li, ['holder'] * len(h_li))) + list(
                        zip(t_li, ['target'] * len(t_li))):
                    start = min(span)
                    end = max(span)
                    if (start, end) in span_startend2tid:
                        tid = span_startend2tid[(start, end)]
                    else:
                        start_off = tind2cind[start] - len(prev_tokens[start]) + sent_offset
                        end_off = tind2cind[end] + sent_offset
                        tid = 'T{}'.format(spanid)
                        ann_file.write('{}\t{} {} {}\t{}\n'.format(
                            tid, label, start_off, end_off, ' '.join(prev_tokens[start:end + 1])))
                        span_startend2tid[(start, end)] = tid
                        spanid += 1
                    tid_label.append((tid, label))

                for i in range(1, len(tid_label)):
                    tid, label = tid_label[i]
                    ann_file.write('{}\t{} {} {}\n'.format('R{}'.format(span_pair_id),
                                                           label,
                                                           'Arg1:{}'.format(tid_label[0][0]),
                                                           'Arg2:{}'.format(tid)))
                    span_pair_id += 1

            sent_count += 1
            sent_offset += len(prev_sent) + 1  # newline

            # new file
            if sent_count >= num_sent_per_doc:
                doc_id += 1
                sent_count, sent_offset, spanid, span_pair_id = 0, 0, 1, 1
                txt_file.close()
                ann_file.close()
                txt_file = ann_file = None

            ds_li, h_li_li, t_li_li = [], [], []

        if sent_str == '':
            break

        sent_seen.add(sent_str)

        ds, h_li, t_li = [], [], []
        ds_start, ds_end = False, False
        bug_sent = False
        prev = 0
        for i, (l, m) in enumerate(zip(labels, mask)):
            if m == 1:
                if ds_end:
                    raise Exception('dup ds')
                ds_start = True
                ds.append(i)
            elif ds_start:
                ds_end = True

            if l == 3:
                h_li.append([])
                h_li[-1].append(i)
            elif l == 4 and prev not in {3, 4}:
                bug_sent = True
                h_li.append([])
                h_li[-1].append(i)
            elif l == 4:
                h_li[-1].append(i)
            elif l == 5:
                t_li.append([])
                t_li[-1].append(i)
            elif l == 6 and prev not in {5, 6}:
                bug_sent = True
                t_li.append([])
                t_li[-1].append(i)
            elif l == 6:
                t_li[-1].append(i)
            prev = l
            '''
            # merge h/t if they are separated
            if l == 3:
                h_li.append([])
                h_li[-1].append(i)
            elif l == 4:
                if len(h_li) == 0:
                    bug_sent = True
                    h_li.append([])
                h_li[-1].append(i)
            elif l == 5:
                t_li.append([])
                t_li[-1].append(i)
            elif l == 6:
                if len(t_li) == 0:
                    bug_sent = True
                    t_li.append([])
                t_li[-1].append(i)
            '''

        prev_sent = sent_str
        sent_str_pred = '-'.join(map(str, ds))
        if sent_str_pred in sent2pred[sent_str]:
            num_dup_pred += 1
            continue
            #raise Exception('duplicate pred')
        sent2pred[sent_str].add(sent_str_pred)

        ds_li.append(ds)
        h_li_li.append(h_li)
        t_li_li.append(t_li)
        num_bug_sent += int(bug_sent)

    if txt_file and not txt_file.closed:
        txt_file.close()
        ann_file.close()

    print('#sent {}, #bug predicate {}, #dup pred {}, #holder {}, #target {}'.format(
        len(sent_seen), num_bug_sent, num_dup_pred, num_holder, num_target))
