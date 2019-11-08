#!/usr/bin/env python

import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser('convert raw format into brat format')
    parser.add_argument('--inp', type=str, required=True, help='input files split by :')
    parser.add_argument('--out', type=str, required=True, help='output dir')
    args = parser.parse_args()

    num_sent_per_doc = 10
    label2pol = {'0': 'neutral', '1': 'positive', '-1': 'negative'}

    ndoc = 0
    ns = 0
    for inp in args.inp.split(':'):
        with open(inp, 'r') as fin:
            while True:
                sent = fin.readline().strip()
                if sent is None or sent == '':
                    break
                target = fin.readline().strip()
                pol = label2pol[fin.readline().strip()]

                if ns % num_sent_per_doc == 0:
                    ndoc += 1
                    if ns > 0:
                        doc_out.close()
                        ann_out.close()
                    doc_out = open(os.path.join(args.out, '{}.txt'.format(ndoc)), 'w')
                    ann_out = open(os.path.join(args.out, '{}.ann'.format(ndoc)), 'w')
                    ns = 0
                    sent_offseet = 0
                    entity_ind = 1

                start = sent.find('$T$')
                end = start + len(target)

                sent = sent[:start] + target + sent[start + 3:]

                doc_out.write('{}\n'.format(sent))
                ann_out.write('{}\t{} {} {}\t{}\n'.format(
                    'T{}'.format(entity_ind), pol, start + sent_offseet, end + sent_offseet, target))

                entity_ind += 1
                sent_offseet += len(sent) + 1
                ns += 1
