from typing import List
from nltk.tree import Tree
import torch
from allennlp.models.constituency_parser import SpanInformation, SpanConstituencyParser
from allennlp.data import Vocabulary

from brat_multitask.dataset_readers.brat import BratDoc


def construct_trees(vocab: Vocabulary,
                    namespace: str,
                    predictions: torch.FloatTensor,
                    all_spans: torch.LongTensor,
                    num_spans: torch.LongTensor,
                    sentences: List[List[str]],
                    pos_tags: List[List[str]] = None) -> List[Tree]:
    """
    Construct ``nltk.Tree``'s for each batch element by greedily nesting spans.
    The trees use exclusive end indices, which contrasts with how spans are
    represented in the rest of the model.
    Parameters
    ----------
    predictions : ``torch.FloatTensor``, required.
        A tensor of shape ``(batch_size, num_spans, span_label_vocab_size)``
        representing a distribution over the label classes per span.
    all_spans : ``torch.LongTensor``, required.
        A tensor of shape (batch_size, num_spans, 2), representing the span
        indices we scored.
    num_spans : ``torch.LongTensor``, required.
        A tensor of shape (batch_size), representing the lengths of non-padded spans
        in ``enumerated_spans``.
    sentences : ``List[List[str]]``, required.
        A list of tokens in the sentence for each element in the batch.
    pos_tags : ``List[List[str]]``, optional (default = None).
        A list of POS tags for each word in the sentence for each element
        in the batch.
    Returns
    -------
    A ``List[Tree]`` containing the decoded trees for each element in the batch.
    """
    # Switch to using exclusive end spans.
    exclusive_end_spans = all_spans.clone()
    exclusive_end_spans[:, :, -1] += 1
    no_label_id = vocab.get_token_index(BratDoc.NEG_SPAN_LABEL, namespace)

    trees: List[Tree] = []
    for batch_index, (scored_spans, spans, sentence) in enumerate(zip(predictions,
                                                                      exclusive_end_spans,
                                                                      sentences)):
        selected_spans = []
        for prediction, span in zip(scored_spans[:num_spans[batch_index]],
                                    spans[:num_spans[batch_index]]):
            start, end = span
            no_label_prob = prediction[no_label_id]
            label_prob, label_index = torch.max(prediction, -1)

            # Does the span have a label != NO-LABEL or is it the root node?
            # If so, include it in the spans that we consider.
            if int(label_index) != no_label_id or (start == 0 and end == len(sentence)):
                # TODO(Mark): Remove this once pylint sorts out named tuples.
                # https://github.com/PyCQA/pylint/issues/1418
                selected_spans.append(SpanInformation(start=int(start), # pylint: disable=no-value-for-parameter
                                                      end=int(end),
                                                      label_prob=float(label_prob),
                                                      no_label_prob=float(no_label_prob),
                                                      label_index=int(label_index)))

        # The spans we've selected might overlap, which causes problems when we try
        # to construct the tree as they won't nest properly.
        consistent_spans = SpanConstituencyParser.resolve_overlap_conflicts_greedily(selected_spans)

        spans_to_labels = {(span.start, span.end): vocab.get_token_from_index(span.label_index, namespace)
                           for span in consistent_spans}
        sentence_pos = pos_tags[batch_index] if pos_tags is not None else None
        trees.append(SpanConstituencyParser.construct_tree_from_spans(spans_to_labels, sentence, sentence_pos))

    return trees
