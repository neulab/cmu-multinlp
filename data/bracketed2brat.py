#!/usr/bin/env python

from typing import Dict, List, Tuple
import argparse
import os
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from nltk.tree import Tree
from tqdm import tqdm


def strip_functional_tags(tree: Tree) -> None:
    """
    Removes all functional tags from constituency labels in an NLTK tree.
    We also strip off anything after a =, - or | character, because these
    are functional tags which we don't want to use.
    This modification is done in-place.
    """
    clean_label = tree.label().split("=")[0].split("-")[0].split("|")[0]
    tree.set_label(clean_label)
    for child in tree:
        if not isinstance(child[0], str):
            strip_functional_tags(child)


def get_gold_spans(tree: Tree, index: int, typed_spans: Dict[Tuple[int, int], str]) -> int:
    """
    Recursively construct the gold spans from an nltk ``Tree``.
    Labels are the constituents, and in the case of nested constituents
    with the same spans, labels are concatenated in parent-child order.
    For example, ``(S (NP (D the) (N man)))`` would have an ``S-NP`` label
    for the outer span, as it has both ``S`` and ``NP`` label.
    Spans are inclusive.
    TODO(Mark): If we encounter a gold nested labelling at test time
    which we haven't encountered, we won't be able to run the model
    at all.
    Parameters
    ----------
    tree : ``Tree``, required.
        An NLTK parse tree to extract spans from.
    index : ``int``, required.
        The index of the current span in the sentence being considered.
    typed_spans : ``Dict[Tuple[int, int], str]``, required.
        A dictionary mapping spans to span labels.
    Returns
    -------
    typed_spans : ``Dict[Tuple[int, int], str]``.
        A dictionary mapping all subtree spans in the parse tree
        to their constituency labels. POS tags are ignored.
    """
    # NLTK leaves are strings.
    if isinstance(tree[0], str):
        # The "length" of a tree is defined by
        # NLTK as the number of children.
        # We don't actually want the spans for leaves, because
        # their labels are POS tags. Instead, we just add the length
        # of the word to the end index as we iterate through.
        end = index + len(tree)
    else:
        # otherwise, the tree has children.
        child_start = index
        for child in tree:
            # typed_spans is being updated inplace.
            end = get_gold_spans(child, child_start, typed_spans)
            child_start = end
        # Set the end index of the current span to
        # the last appended index - 1, as the span is inclusive.
        span = (index, end - 1)
        current_span_label = typed_spans.get(span)
        if current_span_label is None:
            # This span doesn't have nested labels, just
            # use the current node's label.
            typed_spans[span] = tree.label()
        else:
            # This span has already been added, so prepend
            # this label (as we are traversing the tree from
            # the bottom up).
            typed_spans[span] = tree.label() + "-" + current_span_label
    return end


if __name__ == '__main__':
    parser = argparse.ArgumentParser('convert bracketed format into brat format')
    parser.add_argument('--inp', type=str, required=True, help='input file')
    parser.add_argument('--out', type=str, required=True, help='output file')
    parser.add_argument('--num_sent', type=int, default=10)
    args = parser.parse_args()

    use_pos_tags = True
    num_sent_per_doc = args.num_sent

    doc_id, sent_count, sent_offset, spanid = 1, 0, 0, 1
    txt_file = bracket_file = ann_file = None
    directory, filename = os.path.split(args.inp)
    with open(args.inp, 'r') as fin:
        for sid, parse in tqdm(enumerate(BracketParseCorpusReader(root=directory, fileids=[filename]).parsed_sents())):
            if txt_file is None:
                txt_file = open(os.path.join(args.out, '{}.txt'.format(doc_id)), 'w')
                bracket_file = open(os.path.join(args.out, '{}.bracket'.format(doc_id)), 'w')
                ann_file = open(os.path.join(args.out, '{}.ann'.format(doc_id)), 'w')

            strip_functional_tags(parse)
            # This is un-needed and clutters the label space.
            # All the trees also contain a root S node.
            if parse.label() == "VROOT" or parse.label() == "TOP":
                parse = parse[0]
            pos_tags = [x[1] for x in parse.pos()] if use_pos_tags else None

            # tokens
            tokens: List[str] = parse.leaves()
            # spans
            gold_spans: Dict[Tuple[int, int], str] = {}  # both inclusive
            get_gold_spans(parse, 0, gold_spans)
            # raw bracketed format
            raw_line = fin.readline().strip()

            # token ind to end char ind (exclusive)
            tind2cind: Dict[int, int] = {}
            for i, tok in enumerate(tokens):
                if i == 0:
                    tind2cind[i] = len(tok)
                else:
                    tind2cind[i] = tind2cind[i - 1] + len(tok) + 1  # space

            # write sentence and raw bracketed format
            sent = ' '.join(tokens)
            txt_file.write('{}\n'.format(sent))
            bracket_file.write('{}\n'.format(raw_line))
            # write spans
            for (start, end), label in gold_spans.items():
                start_off = tind2cind[start] - len(tokens[start]) + sent_offset
                end_off = tind2cind[end] + sent_offset
                ann_file.write('{}\t{} {} {}\t{}\n'.format(
                    'T{}'.format(spanid), label, start_off, end_off, ' '.join(tokens[start:end + 1])))
                spanid += 1
            sent_count += 1
            sent_offset += len(sent) + 1  # newline

            # new file
            if sent_count >= num_sent_per_doc:
                doc_id += 1
                sent_count, sent_offset, spanid = 0, 0, 1
                txt_file.close()
                bracket_file.close()
                ann_file.close()
                txt_file = bracket_file = ann_file = None

        if txt_file and not txt_file.closed:
            txt_file.close()
            bracket_file.close()
            ann_file.close()
