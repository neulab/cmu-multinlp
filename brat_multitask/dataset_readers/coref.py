import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ListField, TextField, SpanField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, enumerate_spans
from allennlp.data.dataset_readers.coreference_resolution import ConllCorefReader


@DatasetReader.register('my_coref')
class MyConllCorefReader(ConllCorefReader):
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super(MyConllCorefReader, self).__init__(max_span_width, token_indexers, lazy)


    @overrides
    def text_to_instance(self,  # type: ignore
                         sentences: List[List[str]],
                         gold_clusters: Optional[List[List[Tuple[int, int]]]] = None) -> Instance:
        flattened_sentences = [word
                               for sentence in sentences
                               for word in sentence]

        metadata: Dict[str, Any] = {"original_text": flattened_sentences}
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters

        text_field = TextField([Token(word) for word in flattened_sentences], self._token_indexers)

        cluster_dict = {}
        if gold_clusters is not None:
            for cluster_id, cluster in enumerate(gold_clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

        spans: List[Field] = []
        span_labels: Optional[List[int]] = [] if gold_clusters is not None else None

        '''
        sentence_offset = 0
        for sentence in sentences:
            for start, end in enumerate_spans(sentence,
                                              offset=sentence_offset,
                                              max_span_width=self._max_span_width):
                if span_labels is not None:
                    if (start, end) in cluster_dict:
                        span_labels.append(cluster_dict[(start, end)])
                    else:
                        span_labels.append(-1)

                spans.append(SpanField(start, end, text_field))
            sentence_offset += len(sentence)
        '''
        for start, end in enumerate_spans(flattened_sentences,
                                          offset=0,
                                          max_span_width=self._max_span_width):
            if span_labels is not None:
                if (start, end) in cluster_dict:
                    span_labels.append(cluster_dict[(start, end)])
                else:
                    span_labels.append(-1)
            spans.append(SpanField(start, end, text_field))

        spans_ind = sorted(range(len(spans)), key=lambda i: (spans[i].span_end, spans[i].span_start))
        spans = [spans[i] for i in spans_ind]
        span_labels = [span_labels[i] for i in spans_ind]

        span_field = ListField(spans)
        metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {"text": text_field,
                                    "spans": span_field,
                                    "metadata": metadata_field}
        if span_labels is not None:
            fields["span_labels"] = SequenceLabelField(span_labels, span_field)

        return Instance(fields)
