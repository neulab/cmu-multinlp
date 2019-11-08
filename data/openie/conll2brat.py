from typing import Dict, List, Tuple

import argparse
import os
from collections import defaultdict
from tqdm import tqdm

from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence


def conll_oie_to_brat(tokens: List[str],
                      tags_li: List[List[str]],
                      offset: int = 0) -> Tuple[str, Dict[str, Tuple[str, int, int, str]],
                                                Dict[Tuple[str, str], str]]:
    brat_spans: Dict[str, Tuple[str, int, int, str]] = {}
    brat_span_pairs: Dict[Tuple[str, str], str] = {}

    # get token end index
    tokens_len = [len(token) for token in tokens]
    for i in range(1, len(tokens_len)):
        tokens_len[i] += tokens_len[i - 1] + 1  # for space

    for tags in tags_li:  # multiple tag sequences for the same sentence
        # read tokens and get SRL spans
        spans: List[Tuple[str, List[int]]] = []
        span: Tuple[str, List[int]] = (None, [])
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            if tag.startswith('B-'):
                if span[0]:
                    spans.append(span)
                span = (tag[2:], [i])
            elif tag.startswith('I-'):
                if not span[0] or span[0] != tag[2:]:
                    raise Exception('I does not follow a correct B')
                span[1].append(i)
            elif tag == 'O':
                if span[0]:
                    spans.append(span)
                    span = (None, [])
            else:
                raise Exception('invalid BIO tags')
        if span[0]:
            spans.append(span)

        # maks sure there is only one verb-related span
        # merge multiple verb-related spans if any (such as BV and AV used in RnnOIE)
        v_span = ('V', [])
        for span in spans:
            if span[0].endswith('V'):
                if len(v_span[1]) > 0 and v_span[1][-1] != span[1][0] - 1:
                    raise Exception('verb-related spans are not successive to each other')
                v_span[1].extend(span[1])
        # replace all verb-related spans with the new one
        spans = [span for span in spans if not span[0].endswith('V')] + [v_span]

        # convert OIE spans to brat spans
        predicate, args = None, set()
        for i, (label, ind) in enumerate(spans):
            start, end = ind[0], ind[-1]
            token_str = ' '.join(tokens[start:end + 1])
            start = (tokens_len[start - 1] + 1 if start > 0 else 0) + offset
            end = tokens_len[end] + offset
            type = 'Predicate' if label == 'V' else 'Argument'
            key = type + '-' + str(start) + '-' + str(end)
            brat_spans[key] = (type, start, end, token_str)
            if type == 'Predicate':  # keep track of the predicate
                if predicate is not None:
                    raise Exception('multiple predicates in the same annotation')
                predicate = key  # keep track of args
            else:
                args.add((key, label))

        # get brat span pairs (relations)
        if predicate and args:  # skip tag sequences with 'O'
            for arg_key, arg_label in args:
                brat_span_pairs[(predicate, arg_key)] = arg_label

    return ' '.join(tokens), brat_spans, brat_span_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('convert conll 2012 format into brat format')
    parser.add_argument('--inp', type=str, required=True, help='input dir')
    parser.add_argument('--out', type=str, required=True, help='output dir')
    parser.add_argument('--merge', action='store_true', help='merge adjacent same sentences')
    args = parser.parse_args()

    ontonotes_reader = Ontonotes()

    print('reading OpenIE instances from dataset files at: {}. The same sentences must be successive'.format(args.inp))

    def doc_iter(n_sent):  # treat every n_sent sentence as a document for OpenIE to reduce the number of files
        doc: List[OntonotesSentence] = []
        for conll_file in ontonotes_reader.dataset_path_iterator(args.inp):
            for sent in ontonotes_reader.sentence_iterator(conll_file):
                same_as_last = False
                if args.merge and len(doc) > 0 and ' '.join(sent.words) == ' '.join(doc[-1].words):
                    same_as_last = True
                if len(doc) >= n_sent and not same_as_last:
                    yield doc
                    doc = []
                if same_as_last:  # merge same sentences
                    doc[-1].srl_frames.extend(sent.srl_frames)
                else:  # add a new sentence
                    doc.append(sent)
            if len(doc) > 0:
                yield doc
                doc = []

    for docid, doc in tqdm(enumerate(doc_iter(n_sent=10))):
        offset = 0
        docid += 1
        span_count, span_pair_count = 1, 1
        with open(os.path.join(args.out, '{}.txt'.format(docid)), 'w') as doc_out, \
                open(os.path.join(args.out, '{}.ann'.format(docid)), 'w') as ann_out:
            for sent in doc:
                tokens: List[str] = sent.words
                if not sent.srl_frames:
                    # Sentence contains no predicates.
                    tags_li = [['O' for _ in tokens]]
                else:
                    tags_li = []
                    for (_, tags) in sent.srl_frames:
                        tags_li.append(tags)

                # get spans and span_pairs
                sent_str, spans, span_pairs = conll_oie_to_brat(tokens, tags_li, offset=offset)
                offset += len(sent_str) + 1  # newline

                # write sentence to files
                doc_out.write(sent_str + '\n')

                # write span annotations
                span_key_to_ann_key = {}
                span_pair_key_to_ann_key = {}
                for k, v in spans.items():
                    label, start, end, token_str = v
                    span_key_to_ann_key[k] = 'T{}'.format(span_count)
                    span_count += 1
                    ann_out.write('{}\t{} {} {}\t{}\n'.format(span_key_to_ann_key[k], label, start, end, token_str))

                # write span pair annotations
                for (k1, k2), label in span_pairs.items():
                    k = 'R{}'.format(span_pair_count)
                    span_pair_count += 1
                    ann_out.write('{}\t{} {} {}\n'.format(k,
                                                          label,
                                                          'Arg1:{}'.format(span_key_to_ann_key[k1]),
                                                          'Arg2:{}'.format(span_key_to_ann_key[k2])))
