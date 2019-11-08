from typing import Dict, List, Tuple

import argparse
import os
from collections import defaultdict
from tqdm import tqdm

from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser('convert conll 2012 format into brat format')
    parser.add_argument('--inp', type=str, required=True, help='input dir')
    parser.add_argument('--out', type=str, required=True, help='output file')
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

    with open(args.out, 'w') as fout:
        for docid, doc in tqdm(enumerate(doc_iter())):
            for sent in doc:
                if sent.parse_tree is None:
                    continue  # skip bad ann
                    #print(sent.document_id, sent.sentence_id, sent.words)
                fout.write(sent.parse_tree.pformat(margin=1e+10) + '\n')
