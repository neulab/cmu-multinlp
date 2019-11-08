import logging
from typing import Dict, List, Tuple, Iterable, Union, Any, Callable
import os

from overrides import overrides
import spacy
import numpy as np
from random import shuffle
from itertools import combinations
from operator import itemgetter
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from nltk.tree import Tree

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import Field, TextField, SpanField, ListField, SequenceLabelField, \
    ArrayField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)


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


def get_trees_from_bracket_file(filename) -> List[Tree]:
    directory, filename = os.path.split(filename)
    trees = list(BracketParseCorpusReader(root=directory, fileids=[filename]).parsed_sents())
    modified_trees = []
    for tree in trees:
        strip_functional_tags(tree)
        # This is un-needed and clutters the label space.
        # All the trees also contain a root S node.
        if tree.label() == "VROOT" or tree.label() == "TOP":
            tree = tree[0]
        modified_trees.append(tree)
    return modified_trees


class MyToken:
    def __init__(self, text, idx):
        self.text = text
        self.idx = idx


    def __str__(self):
        return str((self.text, self.idx))


    def __repr__(self):
        return str((self.text, self.idx))


def str2bool(text):
    text = text.lower()
    if text == 'true':
        return True
    elif text == 'false':
        return False
    raise Exception('cannot be converted to bool')


def adjust_tokens_wrt_char_boundary(tokens: List[Token], char_boundaries: List[int]):
    '''
    positions indicated by char_boundaries should be segmented.
    If one of the indices is 3, it mean that there is a boundary between the 3rd and 4th char.
    Indices in char_boundaries should be in ascending order.
    '''
    new_tokens: List[MyToken] = []
    cb_ind = 0
    for tok in tokens:
        start = tok.idx
        end = tok.idx + len(tok.text)
        ext_bd = []
        while cb_ind < len(char_boundaries) and char_boundaries[cb_ind] <= end:
            bd = char_boundaries[cb_ind]
            if bd != start and bd != end:  # boundary not detected by tokenizer
                ext_bd.append(bd)
            cb_ind += 1
        for s, e in zip([start] + ext_bd, ext_bd + [end]):
            text = tok.text[s - start:e - start]
            new_tokens.append(MyToken(text, s))
    return new_tokens


class BratDoc:
    EVENT_JOIN_SYM = '->'
    NEG_SPAN_LABEL = '<NEG_SPAN>'
    NEG_SPAN_PAIR_LABEL = '<NEG_SPAN_PAIR>'

    def __init__(self,
                 id: str,
                 doc: Union[str, List[str]],
                 # span_id -> (span_label, start_ind, end_ind),
                 # where start_ind is inclusive and end_ind is exclusive
                 spans: Dict[str, Tuple[str, int, int]],
                 # (span_id1, span_id2) -> span_pair_label
                 span_pairs: Dict[Tuple[str, str], str],
                 bracket_file: str = None,
                 tree: Tree = None):
        self.id = id
        self.doc = doc  # can be str of chars or a list of tokens
        self.spans = spans
        self.span_pairs = span_pairs
        self.bracket_file = bracket_file
        self.tree = tree


    def get_span_weights(self) -> Dict[str, float]:
        ''' compute the weight of the span by how many times it appears in span pairs '''
        span2count: Dict[str, int] = defaultdict(lambda: 0)
        for sid1, sid2 in self.span_pairs:
            span2count[sid1] += 1
            span2count[sid2] += 1
        return dict((k, float(span2count[k])) for k in self.spans)  # can be zero


    def skip_span_pairs(self, labels: set):
        self.span_pairs = dict((k, v) for k, v in self.span_pairs.items() if v not in labels)


    def skip_span(self, labels: set):
        self.spans = dict((k, v) for k, v in self.spans.items() if v not in labels)


    def remove_span_not_in_pair(self):
        sp_set = set(k_ for k in self.span_pairs for k_ in k)
        self.spans = dict((k, v) for k, v in self.spans.items() if k in sp_set)


    def filter(self, max_span_width: int = None):
        ''' remove spans longer than max_span_width '''
        if max_span_width is None:
            return
        new_spans = {}
        for sid, (slabel, sind, eind) in self.spans.items():
            if eind - sind <= max_span_width:
                new_spans[sid] = (slabel, sind, eind)
        new_span_pairs = {}
        for (sid1, sid2), slabel in self.span_pairs.items():
            if sid1 in new_spans and sid2 in new_spans:
                new_span_pairs[(sid1, sid2)] = slabel
        self.spans = new_spans
        self.span_pairs = new_span_pairs


    def truncate(self, max_doc_len):
        ''' truncate the document '''
        # if doc is list of tokens, max_doc_len is the number of tokens to keep
        # if doc is str, max_doc_len is the number of characters to keep
        self.doc = self.doc[:max_doc_len]
        new_spans = {}
        for sid, (slabel, sind, eind) in self.spans.items():
            if sind >= max_doc_len or eind > max_doc_len:
                continue
            new_spans[sid] = (slabel, sind, eind)
        new_span_pairs = {}
        for (sid1, sid2), slabel in self.span_pairs.items():
            if sid1 in new_spans and sid2 in new_spans:
                new_span_pairs[(sid1, sid2)] = slabel
        self.spans = new_spans
        self.span_pairs = new_span_pairs


    def build_cluster(self, inclusive=False) -> List[List[Tuple[int, int]]]:
        cluster: Dict[Tuple[int, int], int] = {}
        num_clusters = 0
        num_overlap_pairs = 0
        for k1, k2 in self.span_pairs:
            offset = 1 if inclusive else 0
            span_parent = (self.spans[k1][1], self.spans[k1][2] - offset)
            span_child = (self.spans[k2][1], self.spans[k2][2] - offset)
            if self.spans[k1][1] < self.spans[k2][2]:
                num_overlap_pairs += 1
            if span_child not in cluster and span_parent not in cluster:
                cluster[span_child] = num_clusters
                cluster[span_parent] = num_clusters
                num_clusters += 1
            elif span_child in cluster and span_parent in cluster:
                if cluster[span_parent] != cluster[span_child]:  # merge
                    from_clu = cluster[span_parent]
                    to_clu = cluster[span_child]
                    for k in cluster:
                        if cluster[k] == from_clu:
                            cluster[k] = to_clu
            elif span_child in cluster:
                cluster[span_parent] = cluster[span_child]
            elif span_parent in cluster:
                cluster[span_child] = cluster[span_parent]
        result = defaultdict(list)
        for k, v in cluster.items():
            result[v].append(k)
        return list(result.values())


    def enumerate_spans(self, *args, **kwargs):
        for start, end in enumerate_spans(self.doc, *args, **kwargs):
            yield start, end + 1  # enumerate_spans is inclusive


    def get_all_neg_spans(self, max_span_width: int = None) -> Dict[str, Tuple[str, int, int]]:
        ''' get all negative spans '''
        pos_span_poss = set((v[1], v[2]) for k, v in self.spans.items())
        if type(self.doc) is not list:
            raise Exception('doc must be tokenized before getting all spans')
        neg_spans = {}
        for start, end in self.enumerate_spans(offset=0, max_span_width=max_span_width):
            if (start, end) not in pos_span_poss:
                # 'TN' for 'T' and negative
                neg_spans['TN' + str(len(neg_spans) + 1)] = (self.NEG_SPAN_LABEL, start, end)
        return neg_spans


    def get_negative_spans_and_span_pairs(self,
                                          neg_ratio: float,
                                          max_span_width: int = None,
                                          neg_spans: Dict[str, Tuple[str, int, int]] = None) \
            -> Tuple[Dict[str, Tuple[str, int, int]], Dict[Tuple[str, str], str]]:
        '''
        Get negatvie spans and span pairs and the number is proportional to the number of words.
        When neg_spans is provided, directly use it to generate negative pairs.
        '''
        seq_len = len(self.doc)
        # At least one negative example. This is for special cases where the sentences are really short
        num_neg = max(1, int(neg_ratio * seq_len))

        if neg_spans is None:
            # generate negative spans
            pos_span_poss = set((v[1], v[2]) for k, v in self.spans.items())
            if type(self.doc) is not list:
                raise Exception('doc must be tokenized before generating negative samples')
            all_spans = list(self.enumerate_spans(offset=0, max_span_width=max_span_width))
            shuffle(all_spans)
            neg_spans = {}
            for start, end in all_spans:
                if (start, end) not in pos_span_poss:
                    if len(neg_spans) >= num_neg:
                        break
                    # 'TN' for 'T' and negative
                    neg_spans['TN' + str(len(neg_spans) + 1)] = (self.NEG_SPAN_LABEL, start, end)

        # generate negative span pairs
        pos_span_pair_ids = set(self.span_pairs.keys())
        all_span_ids = list(self.spans.keys()) + list(neg_spans.keys())
        neg_span_pairs = {}
        used = set()
        def comb():
            for i in range(len(all_span_ids) * len(all_span_ids)):  # n^2 iterations at most
                r1, r2 = np.random.randint(0, len(all_span_ids), 2)
                yield all_span_ids[r1], all_span_ids[r2]
        for s1, s2 in comb():
            if (s1, s2) not in pos_span_pair_ids and (s1, s2) not in used:
                if len(neg_span_pairs) >= num_neg:
                    break
                neg_span_pairs[(s1, s2)] = self.NEG_SPAN_PAIR_LABEL
                used.add((s1, s2))
        return neg_spans, neg_span_pairs


    def to_word(self, tokenizer):
        ''' segment doc and convert char-based index to word-based index '''
        # tokenize
        toks = tokenizer(self.doc)
        char_bd = set()
        for sid, (slabel, start, end) in self.spans.items():
            char_bd.add(start)
            char_bd.add(end)
        toks = adjust_tokens_wrt_char_boundary(toks, char_boundaries=sorted(char_bd))
        words = [tok.text for tok in toks]  # TODO: add normalization?
        # build char ind to token ind mapping
        idxs = [(tok.idx, tok.idx + len(tok.text)) for tok in toks]
        sidx2tidx = dict((s[0], i) for i, s in enumerate(idxs))  # char start ind -> token ind
        eidx2tidx = dict((s[1], i) for i, s in enumerate(idxs))  # char end ind -> token ind
        # convert spans
        new_spans = {}
        for sid, (span_label, sidx, eidx) in self.spans.items():
            if sidx in sidx2tidx and eidx in eidx2tidx:
                new_spans[sid] = (span_label, sidx2tidx[sidx], eidx2tidx[eidx] + 1)  # end index is exclusive
            else:  # remove blanks and re-check
                span_str = self.doc[sidx:eidx]
                blank_str = len(span_str) - len(span_str.lstrip())
                blank_end = len(span_str) - len(span_str.rstrip())
                sidx += blank_str
                eidx -= blank_end
                if sidx in sidx2tidx and eidx in eidx2tidx:
                    new_spans[sid] = (span_label, sidx2tidx[sidx], eidx2tidx[eidx] + 1)  # end index is exclusive
                else:  # the annotation boundary is not consistent with the tokenization boundary
                    raise Exception
        # convert span pairs
        new_span_pairs = dict(((s1, s2), v) for (s1, s2), v in self.span_pairs.items()
                              if s1 in new_spans and s2 in new_spans)
        return BratDoc(self.id, words, new_spans, new_span_pairs, bracket_file=self.bracket_file, tree=self.tree)


    def split_by_sentence(self, sentencizer=None) -> List:
        ''' split into multiple docs by sentence boundary '''
        sents = list(sentencizer(self.doc))  # sentencizer should return the offset between two adjacent sentences

        # split bracket file
        if self.bracket_file:
            trees = get_trees_from_bracket_file(self.bracket_file)
            assert len(trees) == len(sents), '#sent not equal to #tree'

        # collect spans for each sentence
        spans_ord = sorted(self.spans.items(), key=lambda x: (x[1][1], x[1][2]))  # sorted by start ind and end ind
        num_skip_char = 0
        span_ind = 0
        spans_per_sent = []
        for i, (sent, off) in enumerate(sents):
            num_skip_char += off
            spans_per_sent.append([])
            cur_span = spans_per_sent[-1]
            # start ind and end ind should be not larger than a threshold
            while span_ind < len(spans_ord) and \
                    spans_ord[span_ind][1][1] < num_skip_char + len(sent) and \
                    spans_ord[span_ind][1][2] <= num_skip_char + len(sent):
                if spans_ord[span_ind][1][1] < num_skip_char or \
                        spans_ord[span_ind][1][2] <= num_skip_char:
                    logger.warning('span is spreaded across sentences')
                    span_ind += 1
                    continue
                sid, (slabel, sind, eind) = spans_ord[span_ind]
                cur_span.append((sid, (slabel, sind - num_skip_char, eind - num_skip_char)))
                span_ind += 1
            num_skip_char += len(sent)

        # collect span pairs for each sentence
        pair_count = 0
        brat_doc_li = []
        for i, spans in enumerate(spans_per_sent):
            if len(sents[i][0]) <= 0:  # skip empty sentences
                continue
            span_ids = set(span[0] for span in spans)
            span_pair = dict(((s1, s2), v) for (s1, s2), v in self.span_pairs.items()
                             if s1 in span_ids and s2 in span_ids)
            pair_count += len(span_pair)
            tree = trees[i] if self.bracket_file else None
            brat_doc_li.append(BratDoc(self.id, sents[i][0], dict(spans), span_pair, tree=tree))
        # TODO: span pairs across sentences are allowed
        #assert pair_count == len(self.span_pairs), 'some span pairs are spreaded across sentences'

        return brat_doc_li


    @classmethod
    def dummy(cls):
        return cls('id', ['token'], {}, {})


    @classmethod
    def from_file(cls, text_file: str, ann_file: str, bracket_file: str = None):
        ''' read text and annotations from files in BRAT format '''
        with open(text_file, 'r') as txtf:
            doc = txtf.read().rstrip()
        spans = {}
        span_pairs = {}
        eventid2triggerid = {}  # e.g., E10 -> T27
        with open(ann_file, 'r') as annf:
            for l in annf:
                if l.startswith('#'):  # skip comment
                    continue
                if l.startswith('T'):
                    # 1. there are some special chars at the end of the line, so we only strip \n
                    # 2. there are \t in text spans, so we only split twice
                    ann = l.rstrip('\t\n').split('\t', 2)
                else:
                    ann = l.rstrip().split('\t')
                aid = ann[0]
                if aid.startswith('T'):  # text span annotation
                    # TODO: consider non-contiguous span
                    span_label, sind, eind = ann[1].split(';')[0].split(' ')
                    sind, eind = int(sind), int(eind)
                    spans[aid] = (span_label, sind, eind)
                    # TODO: sometime there are spaces, sometimes not, so we cannot assert
                    # sanity check
                    #if len(ann) > 2 and ann[1].find(';') < 0:
                    #    assert ann[2] == doc[sind:eind]
                elif aid.startswith('E'):  # event span annotation
                    events = ann[1].split(' ')
                    trigger_type, trigger_aid = events[0].split(':')
                    eventid2triggerid[aid] = trigger_aid
                    for event in events[1:]:
                        arg_type, arg_aid = event.split(':')
                        span_pairs[(trigger_aid, arg_aid)] = trigger_type + cls.EVENT_JOIN_SYM + arg_type
                elif aid.startswith('R'):  # relation annotation
                    rel = ann[1].split(' ')
                    assert len(rel) == 3
                    rel_type = rel[0]
                    arg1_aid = rel[1].split(':')[1]
                    arg2_aid = rel[2].split(':')[1]
                    span_pairs[(arg1_aid, arg2_aid)] = rel_type
                elif aid.startswith('N'):  # normalization annotation
                    # TODO: how to deal with normalization?
                    pass
                elif not aid[0].istitle():
                    continue  # skip lines not starting with upper case characters
                else:
                    raise NotImplementedError

        # convert event id to text span id
        span_pairs_converted = {}
        for (sid1, sid2), v in span_pairs.items():
            if sid1.startswith('E'):
                sid1 = eventid2triggerid[sid1]
            if sid2.startswith('E'):
                sid2 = eventid2triggerid[sid2]
            span_pairs_converted[(sid1, sid2)] = v
        return cls(ann_file, doc, spans, span_pairs_converted, bracket_file=bracket_file)


    @staticmethod
    def normalize_word(word):
        if word == '/.' or word == '/?':
            return word[1:]
        else:
            return word


class Brat:
    def doc_iter(self, root_dir: str, sentencizer=None):
        '''
        generate a brat doc for each pair of files (.txt and .ann).
        If sentencizer is not None, the document is splitted into several sentences.
        '''
        for root, dirs, files in os.walk(root_dir, followlinks=True):
            for file in files:
                if not file.endswith('.txt'):
                    continue
                text_file = os.path.join(root, file)
                ann_file = os.path.join(root, file[:-4] + '.ann')
                bracket_file = os.path.join(root, file[:-4] + '.bracket')
                if not os.path.exists(ann_file):
                    continue
                if not os.path.exists(bracket_file):
                    bracket_file = None
                brat_doc = BratDoc.from_file(text_file, ann_file, bracket_file=bracket_file)
                if sentencizer:
                    yield from brat_doc.split_by_sentence(sentencizer=sentencizer)
                else:
                    yield brat_doc


@DatasetReader.register('brat')
class BratReader(DatasetReader):
    '''
    Read files in BRAT format.
    The directory should contain pairs of files whose extensions are '.txt' and '.ann' respectively.
    Each text file is a document, and is annotated by chars.
    '''

    PADDING_LABEL = '<PADDING_LABEL>'

    def __init__(self,
                 default_task: str,
                 task_sample_rate: List[int] = None,
                 span_weighted_by_pairs: Dict[str, bool] = None,
                 restart_file: bool = True,
                 use_neg: Dict[str, bool] = None,  # number of negative spans/span pairs per positive one
                 max_span_width: Dict[str, int] = None,  # max number of words in a span
                 max_sent_len: Dict[str, int] = None,  # max length of a sentence
                 max_num_sample: Dict[str, int] = None,  # max number of samples for each task
                 remove_span_not_in_pair: Dict[str, bool] = None,  # e.g., predicates without args in SRL
                 eval_span_pair_skip: Dict[str, List] = None,  # skip these labels during evaluation
                 sentencizer: Dict[str, str] = None,
                 task_sampler: str = None,
                 tokenizer: Dict[str, str] = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super(BratReader, self).__init__(lazy)
        self._default_task = default_task
        self._task_sample_rate = task_sample_rate
        # task name -> whether to use weights for span loss (not use by default)
        self._span_weighted_by_pairs = span_weighted_by_pairs or defaultdict(lambda: False)
        self._restart_file = restart_file
        if use_neg is None:
            use_neg = defaultdict(lambda: True)
        self._use_neg = use_neg
        if max_span_width is None:
            max_span_width = defaultdict(lambda: None)  # default is not restriction
        self._max_span_width = max_span_width
        if max_sent_len is None:
            max_sent_len = defaultdict(lambda: None)  # default not truncate
        self._max_sent_len = max_sent_len
        if max_num_sample is None:
            max_num_sample = defaultdict(lambda: None)  # default no limit
        self._max_num_sample = max_num_sample
        self._task_sampler = task_sampler
        if remove_span_not_in_pair is None:
            remove_span_not_in_pair = defaultdict(lambda: False)
        self._remove_span_not_in_pair = remove_span_not_in_pair
        self._eval_span_pair_skip = None
        if eval_span_pair_skip is not None:
            self._eval_span_pair_skip = defaultdict(set)
            for k, v in eval_span_pair_skip.items():
                self._eval_span_pair_skip[k].update(v)

        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
        self._nlp.add_pipe(self._nlp.create_pipe('sentencizer'))

        # sentencizer
        def sentencizer_spacy(doc):  # TODO: use sentencizer might cause lots of spans across sentences
            n_newline = 0
            for i, _sent in enumerate(doc.split('\n')):
                prev = -n_newline
                sents = self._nlp(_sent)
                sents.is_parsed = True  # spacy's bug
                for j, sent in enumerate(sents.sents):
                    off = sent.start_char - prev
                    prev = sent.end_char
                    yield sent.text, off
                    n_newline = len(_sent) - sent.end_char
                n_newline += 1  # for "\n"
        def sentencizer_newline(doc):
            for i, _sent in enumerate(doc.split('\n')):
                yield _sent, 0 if i == 0 else 1
        def sentencizer_concat(doc):
            yield doc.replace('\n', ' '), 0
        n2sentencizer = {
            'spacy': sentencizer_spacy,
            'newline': sentencizer_newline,
            'concat': sentencizer_concat
        }
        if sentencizer is None:
            self._sentencizer = defaultdict(lambda: sentencizer_newline)
        else:
            self._sentencizer = dict((k, n2sentencizer[v]) for k, v in sentencizer.items())

        # tokenizer
        tokenizer_spacy = lambda sent: self._nlp(sent)
        def tokenizer_space(sent):
            tokens = []
            offset = 0
            for i, t in enumerate(sent.split(' ')):
                tokens.append(MyToken(t, offset))
                offset += len(t) + 1  # for space
            return tokens
        n2tokenizer = {
            'spacy': tokenizer_spacy,
            'space': tokenizer_space
        }
        if tokenizer is None:
            self._tokenizer = defaultdict(lambda: tokenizer_spacy)
        else:
            self._tokenizer: Dict[str, Callable] = dict((k, n2tokenizer[v]) for k, v in tokenizer.items())


    @overrides
    def _read(self, file_path: str):
        task_fp_li = file_path.split(':')  # dirs for multiple tasks are separated by ":"
        task_fp_li = [tfp.split('|', 2) for tfp in task_fp_li]  # each task is of the format "task|e2e|filepath"
        task_fp_li = [(task, str2bool(e2e), file_path) for task, e2e, file_path in task_fp_li]
        is_training = file_path.find('/train') != -1

        task_sample_rate = self._task_sample_rate
        if task_sample_rate is None:  # use uniform sampling as default
            task_sample_rate = [1] * len(task_fp_li)

        assert len(task_sample_rate) == len(task_fp_li), 'length inconsistent'

        # use task sampler instead of specifying sampling rate manually
        if is_training and self._task_sampler:
            if not hasattr(self, '_num_samples'):
                self._num_samples = np.array([list(self._read_one_task(
                    file_path, task=task, e2e=e2e, estimate_count=True))[0]
                                     for task, e2e, file_path in task_fp_li])
                logger.info('#samples estimation: {}'.format(self._num_samples))
            if self._task_sampler == 'log':
                task_sample_rate = self._num_samples / np.min(self._num_samples)
                task_sample_rate = np.log2(task_sample_rate + 1)
            elif self._task_sampler == 'sqrt':
                task_sample_rate = self._num_samples / np.min(self._num_samples)
                task_sample_rate = np.sqrt(task_sample_rate)
            elif self._task_sampler == 'proportional':
                task_sample_rate = self._num_samples / np.min(self._num_samples)
            else:
                raise NotImplemented

        # convert sample rate to distribution
        task_sample_rate = np.array(task_sample_rate, dtype=np.float32)
        task_sample_rate /= task_sample_rate.sum()
        logger.info('task sample rate is: {}'.format(task_sample_rate))

        # initialize all the readers
        readers = [self._read_one_task(file_path, task=task, e2e=e2e) for task, e2e, file_path in task_fp_li]
        stop_set = set()
        restart = 0
        while True:
            # TODO: remove buf
            buf: List[Instance] = []
            # random sample a task
            i = np.random.choice(len(task_sample_rate), 1, p=task_sample_rate)[0]
            reader = readers[i]
            task, e2e, file_path = task_fp_li[i]
            try:
                buf.append(next(reader))
            except StopIteration:
                stop_set.add(i)
                if len(stop_set) >= len(task_fp_li):  # exit when all files are exhausted
                    break
                if is_training and self._restart_file:  # restart the current file only at training
                    restart += 1
                    readers[i] = self._read_one_task(file_path, task=task, e2e=e2e)
                    buf.append(next(readers[i]))
                else:
                    continue
            yield from buf


    def _read_one_task(self,
                       file_path: str,
                       task: str = None,
                       # whether to use end2end setting, where the span pairs are constructed during forward
                       # by selecting the spans with highest scores
                       e2e: bool = False,
                       estimate_count: bool = False) -> Iterable[Instance]:
        file_path = cached_path(file_path)
        is_training = file_path.find('/train') != -1
        brat_reader = Brat()
        task = self._default_task if task is None else task
        max_span_width = self._max_span_width[task]
        ns_before = ns_after = 0  # track number of spans
        nsp_before = nsp_after = 0  # track number of span pairs
        num_samples = 0

        def iter(file_path, sentencizer):
            nonlocal ns_before, nsp_before
            num_sam = 0
            stop = False
            for brat_doc in tqdm(brat_reader.doc_iter(file_path, sentencizer=None), disable=not estimate_count):
                ns_before += len(brat_doc.spans)
                nsp_before += len(brat_doc.span_pairs)
                for brat_sent in brat_doc.split_by_sentence(sentencizer=sentencizer):
                    num_sam += 1
                    yield brat_sent
                    if is_training and self._max_num_sample[task] is not None and num_sam >= self._max_num_sample[task]:
                        stop = True
                        break
                if stop:
                    break

        # split by sentences
        for di, brat_doc in enumerate(iter(file_path, sentencizer=self._sentencizer[task])):

            brat_doc = brat_doc.to_word(self._tokenizer[task])
            raw_tokens = [t for t in brat_doc.doc]
            raw_brat_doc = deepcopy(brat_doc)

            brat_doc.filter(max_span_width)
            if self._max_sent_len[task] is not None:
                brat_doc.truncate(self._max_sent_len[task])
            if not is_training and self._eval_span_pair_skip and self._eval_span_pair_skip[task]:
                # skip some labels during evaluation
                brat_doc.skip_span_pairs(self._eval_span_pair_skip[task])
            if self._remove_span_not_in_pair[task]:
                brat_doc.remove_span_not_in_pair()

            ns_after += len(brat_doc.spans)
            nsp_after += len(brat_doc.span_pairs)

            num_samples += 1
            if estimate_count:
                continue

            tokens = [Token(t) for t in brat_doc.doc]
            span_items = brat_doc.spans.items()
            spans = [(sid, (sind, eind)) for sid, (slabel, sind, eind) in span_items]
            span_labels = [slabel for sid, (slabel, sind, eind) in span_items]
            if task and self._span_weighted_by_pairs[task]:
                span_weights_dict = brat_doc.get_span_weights()
            else:
                span_weights_dict = defaultdict(lambda: 1.0)
            span_weights = [span_weights_dict[sid] for sid, _ in span_items]
            span_pair_items = brat_doc.span_pairs.items()
            span_pairs = list(map(itemgetter(0), span_pair_items))
            span_pair_labels = list(map(itemgetter(1), span_pair_items))

            # Use neg_rate to control whether to include negative spans.
            # Some tasks (e.g., aspect-based sentiment analysis task subtask 2) don't need negative spans.
            # We never need negative span pairs in either of the e2e or non-e2e setting.
            if self._use_neg[task]:
                neg_spans = brat_doc.get_all_neg_spans(max_span_width)

                neg_spans = neg_spans.items()
                spans += [(sid, (sind, eind)) for sid, (slabel, sind, eind) in neg_spans]
                span_labels += [slabel for sid, (slabel, sind, eind) in neg_spans]
                span_weights += [1.0] * len(neg_spans)

            '''
            # Always add negative examples. Whether they are used is determined by the forward function of the model.
            # Sometimes there is not pos examples not because the task doesn't have this loss, but because there is
            # in fact no pos examples, e.g., when all the words are labels 'O' in NER.
            if self._neg_rate and is_training:
                # add all negative spans for training
                neg_spans = brat_doc.get_all_neg_spans(max_span_width)
                # add negative spans pairs proportional to the number of tokens for training
                neg_spans, neg_span_pairs = brat_doc.get_negative_spans_and_span_pairs(
                    self._neg_rate, max_span_width, neg_spans=neg_spans)

                neg_spans = neg_spans.items()
                spans += [(sid, (sind, eind)) for sid, (slabel, sind, eind) in neg_spans]
                span_labels += [slabel for sid, (slabel, sind, eind) in neg_spans]
                span_weights += [1.0] * len(neg_spans)

                neg_span_pairs = neg_span_pairs.items()
                span_pairs += list(map(itemgetter(0), neg_span_pairs))
                span_pair_labels += list(map(itemgetter(1), neg_span_pairs))

            if not is_training:
                # get all neg spans for dev/test but keep span paris unchanged
                # in case that the model want to evaluate on gold spans
                neg_spans = brat_doc.get_all_neg_spans(max_span_width)

                neg_spans = neg_spans.items()
                spans += [(sid, (sind, eind)) for sid, (slabel, sind, eind) in neg_spans]
                span_labels += [slabel for sid, (slabel, sind, eind) in neg_spans]
                span_weights += [1.0] * len(neg_spans)
            '''

            if len(spans) <= 0 and len(span_pairs) <= 0:
                raise Exception('no supervisions are contained by this example')

            yield self.text_to_instance(tokens, spans, span_pairs, span_weights,
                                        task=task, span_labels=span_labels,
                                        span_pair_labels=span_pair_labels, e2e=e2e,
                                        raw_tokens=raw_tokens, real=True, brat_doc=raw_brat_doc,
                                        tree=raw_brat_doc.tree)

        logger.warning('#spans {} -> {} in {}'.format(ns_before, ns_after, file_path))
        logger.warning('#spans pairs {} -> {} in {}'.format(nsp_before, nsp_after, file_path))

        if estimate_count:
            yield num_samples
            return

        # add a fake example to make sure the neg labels are built into the vocab
        # TODO add span pair weigh
        yield self.text_to_instance([Token('token')], [('T1', (0, 1))], [('T1', 'T1')], [0.0],
                                    task=task, span_labels=[BratDoc.NEG_SPAN_LABEL],
                                    span_pair_labels=[BratDoc.NEG_SPAN_PAIR_LABEL], e2e=e2e,
                                    raw_tokens=['token'], real=False, brat_doc=BratDoc.dummy(),
                                    tree=None)


    def text_to_instance(self,
                         tokens: List[Token],
                         spans: List[Tuple[str, Tuple[int, int]]],  # end ind is exclusive
                         span_pairs: List[Tuple[str, str]],
                         span_weights: List[float],
                         task: str = None,
                         span_labels: List[str] = None,
                         span_pair_labels: List[str] = None,
                         e2e: bool = False,
                         **kwargs) -> Instance:
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        # Spans must be ordered by the end index because we might need to
        # remove some of them during forward computation mainly because of
        # the length constraints introduced by models like BERT. In this case,
        # we hope all the removed spans are located at the end of the list.
        spans_ind = sorted(range(len(spans)), key=lambda i: (spans[i][1][1], spans[i][1][0]))
        spans = [spans[i] for i in spans_ind]
        sid2ind = dict((s[0], i) for i, s in enumerate(spans))
        span_field = ListField([SpanField(sind, eind - 1, text_field) for sid, (sind, eind) in spans])
        task = self._default_task if task is None else task
        task_field = LabelField(task, label_namespace='task_labels')
        if len(span_pairs) > 0:
            span_pair_field = ListField([ArrayField(
                np.array([sid2ind[sid1], sid2ind[sid2]], dtype=np.int64),
                padding_value=-1,  # the same as span field
                dtype=np.int64
            ) for sid1, sid2 in span_pairs])
        else:
            span_pair_field = ListField([ArrayField(
                np.array([-1, -1], dtype=np.int64),  # use a padding sample as placeholder
                padding_value=-1,
                dtype=np.int64
            )])
        assert len(spans) == len(span_weights), 'input and weights length inconsistent'
        span_weights = [span_weights[i] for i in spans_ind]  # to be consistent with sorted spans
        span_weights_field = ArrayField(np.array(span_weights, dtype=np.float32), padding_value=0, dtype=np.float32)
        fields: Dict[str, Field] = {
            'text': text_field,
            'spans': span_field,
            'task_labels': task_field,
            'span_pairs': span_pair_field,
            'span_weights': span_weights_field
        }
        if span_labels is not None:
            # TODO debug (consti label transformation)
            '''
            def consti_map(sp):
                if sp == 'NP' or sp == BratDoc.NEG_SPAN_LABEL:
                    return sp
                return 'S'
            if task == 'consti':
                span_labels = [consti_map(sp) for sp in span_labels]
            '''
            assert len(spans) == len(span_labels), 'input and label length inconsistent'
            span_labels = [span_labels[i] for i in spans_ind]  # to be consistent with sorted spans
            fields['span_labels'] = SequenceLabelField(
                span_labels, span_field, label_namespace='{}_span_labels'.format(task))
        if span_pair_labels is not None:
            if len(span_pairs) > 0:
                assert len(span_pairs) == len(span_pair_labels), 'input and label length inconsistent'
                fields['span_pair_labels'] = SequenceLabelField(
                    span_pair_labels, span_pair_field, label_namespace='{}_span_pair_labels'.format(task))
            else:
                fields['span_pair_labels'] = SequenceLabelField(
                    [self.PADDING_LABEL], span_pair_field, label_namespace='{}_span_pair_labels'.format(task))
        # add meta filed
        # e2e is used in forward to decide whether to use end2end training/testing
        metadata_dict: Dict[str, Any] = {'task': task, 'e2e': e2e}
        if 'brat_doc' in kwargs:
            metadata_dict['clusters'] = kwargs['brat_doc'].build_cluster(inclusive=True)
        metadata_dict.update(kwargs)
        fields['metadata'] = MetadataField(metadata_dict)
        return Instance(fields)
