from typing import Dict, List, Tuple

import argparse
import os
from collections import defaultdict
from tqdm import tqdm

from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser('convert conll 2012 format into brat format')
    parser.add_argument('--inp', type=str, required=True, help='input dir')
    parser.add_argument('--out', type=str, required=True, help='output dir')
    parser.add_argument('--filter', type=str, default=None, help='doc ids used to filter ontonotes datasets')
    parser.add_argument('--domain', type=str, default=None, help='domain identifier')
    args = parser.parse_args()

    ontonotes_reader = Ontonotes()

    print('reading SRL instances from dataset files at: {}'.format(args.inp))
    if args.domain is not None:
        print('only include file paths containing the {} domain'.format(args.domain))

    docids = set()
    if args.filter:
        with open(args.filter, 'r') as fin:
            for line in fin:
                docid = line.strip().split('annotations/')[1]
                docids.add(docid)
        print('#docs in filter set: {}'.format(len(docids)))

    def doc_iter():
        for conll_file in ontonotes_reader.dataset_path_iterator(args.inp):
            docid = conll_file.split('annotations/')[1].rsplit('.', 1)[0]
            if args.filter and docid not in docids:
                continue
            if args.domain is None or f'/{args.domain}/' in conll_file:
                yield from ontonotes_reader.dataset_document_iterator(conll_file)

    for docid, doc in tqdm(enumerate(doc_iter())):
        offset = 0
        docid += 1
        span_count = 1
        with open(os.path.join(args.out, '{}.txt'.format(docid)), 'w') as doc_out, \
                open(os.path.join(args.out, '{}.ann'.format(docid)), 'w') as ann_out:
            for sent in doc:
                tokens: List[str] = sent.words
                pos: List[str] = sent.pos_tags

                assert len(tokens) == len(pos)

                # write sentence to files
                sent_str = ' '.join(tokens)
                doc_out.write(sent_str + '\n')

                # write span annotations
                end = offset - 1
                for t, p in zip(tokens, pos):
                    if t.find(' ') != -1:
                        raise Exception('space in token')
                    start, end = end + 1, end + 1 + len(t)  # space
                    ann_out.write('{}\t{} {} {}\t{}\n'.format('T{}'.format(span_count), p, start, end, t))
                    span_count += 1

                offset += len(sent_str) + 1  # newline
