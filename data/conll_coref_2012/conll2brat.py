# based on https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/coreference_resolution/conll.py

from typing import Dict, List, Tuple, DefaultDict

import argparse
import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence
from allennlp.data.dataset_readers.coreference_resolution.conll import canonicalize_clusters


def cluster_to_brat(doc: List[List[str]], clusters: List[List[Tuple[int, int]]]) -> Tuple[str, Dict, set]:
    # word index -> start/end char index (use whitespace bewtween words and newline between sentences)
    word2char: Dict[int, Tuple[int, int]] = {}
    word_ind = 0
    char_ind = 0
    for i, sent in enumerate(doc):
        if i > 0:
            char_ind += 1  # newline
        for j, w in enumerate(sent):
            if j > 0:
                char_ind += 1  # whitespace
            word2char[word_ind] = (char_ind, char_ind + len(w))
            word_ind += 1
            char_ind += len(w)

    # replace word ind with char ind in clusters
    clusters = [[(word2char[m[0]][0], word2char[m[1]][1]) for m in cluster] for cluster in clusters]

    # get span and span pairs
    doc_str = '\n'.join(' '.join(s) for s in doc)
    spans: Dict[Tuple, str] = defaultdict(lambda: 'T{}'.format(len(spans) + 1))
    span_pairs = set()
    for cluster in clusters:
        # sort all the mentions according to their order in the document
        cluster = sorted(cluster, key=lambda x: (x[0], x[1]))
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                # j -> i
                i_ind = spans[cluster[i]]
                j_ind = spans[cluster[j]]
                span_pairs.add((j_ind, i_ind))
    return doc_str, spans, span_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('convert conll 2012 format into brat format')
    parser.add_argument('--inp', type=str, required=True, help='input dir')
    parser.add_argument('--out', type=str, required=True, help='output dir')
    args = parser.parse_args()

    print('reading coref instances from dataset files at: {}'.format(args.inp))

    avg_cluster_size = []
    ontonotes_reader = Ontonotes()
    for docid, doc in tqdm(enumerate(ontonotes_reader.dataset_document_iterator(args.inp))):
        docid += 1
        clusters: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)

        total_tokens = 0
        for sentence in doc:
            for typed_span in sentence.coref_spans:
                span_id, (start, end) = typed_span  # both start and end are inclusive
                clusters[span_id].append((start + total_tokens, end + total_tokens))
            total_tokens += len(sentence.words)

        canonical_clusters = canonicalize_clusters(clusters)
        avg_cluster_size.extend([len(c) for c in canonical_clusters])
        doc_str, spans, span_pairs = cluster_to_brat([s.words for s in doc], canonical_clusters)

        with open(os.path.join(args.out, '{}.txt'.format(docid)), 'w') as doc_out, \
                open(os.path.join(args.out, '{}.ann'.format(docid)), 'w') as ann_out:
            doc_out.write('{}\n'.format(doc_str))

            for v, k in spans.items():
                start, end = v
                token_str = doc_str[start:end]
                ann_out.write('{}\t{} {} {}\t{}\n'.format(k, 'mention', start, end, token_str))

            # write span pair annotations
            for i, (k1, k2) in enumerate(span_pairs):
                ann_out.write('{}\t{} {} {}\n'.format('R{}'.format(i + 1), 'coreference',
                                                      'Arg1:{}'.format(k1), 'Arg2:{}'.format(k2)))
    print('avg size of clusters: {}, totally {}'.format(np.mean(avg_cluster_size), len(avg_cluster_size)))
