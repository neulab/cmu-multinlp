from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Set

from overrides import overrides
import torch
import torch.nn as nn
from torch.nn.modules import Linear
import torch.nn.functional as F
import numpy as np
import math
from collections import defaultdict
from copy import deepcopy
import time

from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder, TimeDistributed, Embedding
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from allennlp.models.model import Model
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import sequence_cross_entropy_with_logits, logsumexp
from allennlp.training.metrics import MentionRecall, EvalbBracketingScorer, DEFAULT_EVALB_DIR

from brat_multitask.modules import SpanPairLayer, SpanPairPairedLayer, SpanPairLabelProjectionLayer, \
    HeadSelfAttentiveSpanExtractor, HighOrderLayer, BertSelfAttnLayers, BasicTextFieldEmbedder
from brat_multitask.modules.bert_token_embedder import BertEmbedder
from brat_multitask.modules.constituency_parsing import construct_trees
from brat_multitask.dataset_readers.brat import BratDoc, BratReader
from brat_multitask.metrics import PrecisionRecallF1, MyCategoricalAccuracy, MyConllCorefScores, \
    Semeval2010, MultipleLoss


@Model.register('brat')
class BratMultitask(Model):
    '''
    A multitask IE/NLP model that consumes data in BRAT format and is trained by:
    (1) predicting labels for each span (e.g., NER).
    (2) predicting relations between two spans (e.g., relation extraction).
    '''
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: BasicTextFieldEmbedder,
                 span_layer: Dict[str, FeedForward],
                 span_pair_layer: Dict[str, SpanPairPairedLayer],
                 context_layer: Seq2SeqEncoder = None,
                 use_context_layer: bool = True,
                 span_repr_combination: str = 'x,y',
                 use_attentive_span_repr: bool = True,
                 use_head_attentive_span_repr: bool = False,
                 attentive_after_context: bool = False,
                 attentive_dim_reduction: int = None,
                 post_attentive_layer: FeedForward = None,
                 span_width_embedding_dim: int = None,
                 num_order: int = 0,
                 max_span_width: int = None,
                 span_pair_prediction_method: str = 'mlp',
                 spans_per_word: Dict[str, str] = None,
                 task_weight: Dict[str, float] = None,
                 task_loss: Dict[str, set] = None,
                 task_loss_reduction: Dict[str, str] = None,
                 truncate_span_loss: Dict[str, bool] = None,
                 special_loss: Dict[str, bool] = None,
                 special_metric: Dict[str, List[str]] = None,
                 combine_span_and_pair_score: Dict[str, bool] = None,
                 pair_ind_method: Dict[str, str] = None,
                 lexical_dropout: float = 0.0,
                 span_label_as_emb: Tuple[str, int] = None,
                 task_emb_dim: int = None,
                 use_uncertainty_weight: bool = False,
                 task_list: List[str] = None,
                 different_span_repr: bool = False,
                 bucket_widths: bool = False,
                 log_sim_file: str = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BratMultitask, self).__init__(vocab, regularizer)

        # ------ shared layers ------
        # word embedding layer (word2ve, ELMo, or BERT)
        self._text_field_embedder = text_field_embedder
        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x

        # span label embedding layer (task name and emb dim, only applicable to single-task learning)
        self._span_label_emb = None
        if span_label_as_emb:
            self._span_label_emb = nn.Embedding(
                self.vocab.get_vocab_size('{}_span_labels'.format(span_label_as_emb[0])),
                span_label_as_emb[1], padding_idx=None)

        '''
        # contextual span repr layer
        self._context_layer = context_layer
        if context_layer is not None:
            self._endpoint_span_extractor = EndpointSpanExtractor(context_layer.get_output_dim(),
                                                                  combination=span_repr_combination,
                                                                  num_width_embeddings=max_span_width,
                                                                  span_width_embedding_dim=span_width_embedding_dim,
                                                                  bucket_widths=bucket_widths)
        # attention span repr layer
        self._attentive_span_extractor = None
        if use_attentive_span_repr:
            attn_inp_dim = text_field_embedder.get_output_dim()
            self._attentive_dim_reduction = None
            if attentive_dim_reduction:
                self._attentive_dim_reduction = TimeDistributed(
                    Linear(text_field_embedder.get_output_dim(), attentive_dim_reduction))
                attn_inp_dim = attentive_dim_reduction
            self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=attn_inp_dim)
            self._post_attentive = lambda x: x
            if post_attentive_layer is not None:
                self._post_attentive = TimeDistributed(post_attentive_layer)
        '''

        if spans_per_word is None:
            spans_per_word = defaultdict(lambda: 1.0)  # default is 1.0
        else:
            spans_per_word = dict((k, eval(v)) for k, v in spans_per_word.items())
        self._spans_per_word = spans_per_word
        if task_weight is None:
            task_weight = defaultdict(lambda: 1.0)
        self._task_weight = task_weight
        if task_loss is None:
            task_loss = defaultdict(lambda: ['span', 'span_pair'])  # default is having two losses
        self._task_loss = task_loss  # specifies the loss of each task
        if task_loss_reduction is None:
            task_loss_reduction = defaultdict(lambda: 'batch')
        self._task_loss_reduction = task_loss_reduction
        # specifies whether to combine span score and span pair score
        if combine_span_and_pair_score is None:
            combine_span_and_pair_score = defaultdict(lambda: False)  # default is not combine
        self._combine_span_and_pair_score = combine_span_and_pair_score
        if special_loss is None:
            special_loss = defaultdict(lambda: False)  # default is not use special loss
        self._special_loss = special_loss
        if special_metric is None:
            special_metric = defaultdict(lambda: [])
        self._special_metric = special_metric
        if pair_ind_method is None:
            pair_ind_method = defaultdict(lambda: None)
        self._pair_ind_method = pair_ind_method
        if truncate_span_loss is None:
            truncate_span_loss = defaultdict(lambda: True)
        self._truncate_span_loss = truncate_span_loss
        self._use_uncertainty_weight = use_uncertainty_weight
        self._task_emb_dim = task_emb_dim
        self._different_span_repr = different_span_repr
        self._num_order = num_order
        self._log_sim_file = log_sim_file
        self._mid_layer = None

        # ------ task-specific layers ------
        self.num_tasks = self.vocab.get_vocab_size('task_labels')
        self.ind2task = self.vocab.get_index_to_token_vocabulary('task_labels').items()

        if task_emb_dim:
            self.task_emb = Embedding(len(self.ind2task), task_emb_dim)

        if len(span_layer) == 1 and len(self.ind2task) > 1:
            same_span_layer = list(span_layer.values())[0]
            for task_ind, task_name in self.ind2task:
                if task_name not in span_layer:
                    span_layer[task_name] = deepcopy(same_span_layer)

        if len(span_pair_layer) == 1 and len(self.ind2task) > 1:
            same_span_pair_layer = list(span_pair_layer.values())[0]
            for task_ind, task_name in self.ind2task:
                if task_name not in span_pair_layer:
                    span_pair_layer[task_name] = deepcopy(same_span_pair_layer)

        if task_list:
            self.ind2task = [(task_ind, task_name) for task_ind, task_name in self.ind2task if task_name in task_list]

        # tract loss of each task separately
        self.multi_loss = MultipleLoss([task_name for task_ind, task_name in self.ind2task])

        # context layer
        if not use_context_layer:
            context_layer = None

        for task_ind, task_name in self.ind2task:
            # contextual span repr layer
            if not different_span_repr:
                self._context_layer = context_layer
            else:
                setattr(self, '_context_layer_{}'.format(task_name), deepcopy(context_layer))
            if context_layer is not None:
                if not different_span_repr:
                    self._endpoint_span_extractor = EndpointSpanExtractor(context_layer.get_output_dim(),
                                                                          combination=span_repr_combination,
                                                                          num_width_embeddings=max_span_width,
                                                                          span_width_embedding_dim=span_width_embedding_dim,
                                                                          bucket_widths=bucket_widths)
                else:
                    setattr(self, '_endpoint_span_extractor_{}'.format(task_name),
                            EndpointSpanExtractor(context_layer.get_output_dim(),
                                                  combination=span_repr_combination,
                                                  num_width_embeddings=max_span_width,
                                                  span_width_embedding_dim=span_width_embedding_dim,
                                                  bucket_widths=bucket_widths))
            # attention span repr layer
            # TODO: add task-specific layers
            self._attentive_span_extractor = None
            self._attentive_after_context = attentive_after_context
            if use_attentive_span_repr:
                if attentive_after_context:
                    attn_inp_dim = context_layer.get_output_dim()
                else:
                    attn_inp_dim = text_field_embedder.get_output_dim()
                self._attentive_dim_reduction = None
                if attentive_dim_reduction:
                    self._attentive_dim_reduction = TimeDistributed(
                        Linear(attn_inp_dim, attentive_dim_reduction))
                    attn_inp_dim = attentive_dim_reduction
                if not different_span_repr:
                    if use_head_attentive_span_repr:
                        self._attentive_span_extractor = HeadSelfAttentiveSpanExtractor(input_dim=attn_inp_dim)
                    else:
                        self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=attn_inp_dim)
                else:
                    setattr(self, '_attentive_span_extractor_{}'.format(task_name),
                            SelfAttentiveSpanExtractor(input_dim=attn_inp_dim))

                if hasattr(self._text_field_embedder, 'token_embedder_bert') and \
                        isinstance(self._text_field_embedder.token_embedder_bert, BertEmbedder) and \
                        self._text_field_embedder.token_embedder_bert.use_middle_layer is not None:
                    # use the BERT layers after the middle layer to represent spans
                    self._mid_layer = self._text_field_embedder.token_embedder_bert.use_middle_layer
                    bert_layers = self._text_field_embedder.token_embedder_bert.bert_model.encoder.layer[self._mid_layer + 1:]
                    bert_layers = BertSelfAttnLayers(bert_layers)
                    self._bert_span_layer = HeadSelfAttentiveSpanExtractor(
                        input_dim=attn_inp_dim, num_head=int(1e10), bert_self_attn_layers=bert_layers)

                self._post_attentive = lambda x: x
                if post_attentive_layer is not None:
                    self._post_attentive = TimeDistributed(post_attentive_layer)

            if not different_span_repr:
                break

        for task_ind, task_name in self.ind2task:
            # span predictor
            span_label_ns = '{}_span_labels'.format(task_name)
            setattr(self, '{}_span_num_classes'.format(task_name), self.vocab.get_vocab_size(span_label_ns))
            setattr(self, '{}_span_neg_label'.format(task_name),
                    self.vocab.get_token_index(BratDoc.NEG_SPAN_LABEL, span_label_ns))
            span_num_classes = getattr(self, '{}_span_num_classes'.format(task_name))
            span_neg_label = getattr(self, '{}_span_neg_label'.format(task_name))
            setattr(self, '{}_span_layer'.format(task_name), TimeDistributed(span_layer[task_name]))
            setattr(self, '{}_span_label_proj'.format(task_name),
                    TimeDistributed(Linear(span_layer[task_name].get_output_dim(), span_num_classes)))

            if num_order:
                setattr(self, '{}_high_order_layer'.format(task_name),
                        HighOrderLayer(input_dim=span_layer[task_name].get_input_dim(),
                                       task2majorlabel={'coref': 'coreference'},
                                       num_order=num_order,
                                       task2dummy={'coref': True},
                                       vocab=vocab))

            if self._use_uncertainty_weight:
                # learnable uncertainty weight log(\delta^2) initialized as zero
                setattr(self, '{}_uncertain_weight'.format(task_name), nn.Parameter(torch.tensor(0.0)))

            if 'span' in self._task_loss[task_name]:
                # metrics
                setattr(self, '{}_s_acc'.format(task_name), MyCategoricalAccuracy(top_k=1, tie_break=False))
                setattr(self, '{}_s_prf'.format(task_name), PrecisionRecallF1(neg_label=span_neg_label))
                setattr(self, '{}_s_prf_b'.format(task_name),
                        PrecisionRecallF1(neg_label=span_neg_label, binary_match=True))

            if 'span_pair' in self._task_loss[task_name]:
                # span pair predictor
                span_pair_label_ns = '{}_span_pair_labels'.format(task_name)
                setattr(self, '{}_span_pair_num_classes'.format(task_name), self.vocab.get_vocab_size(span_pair_label_ns))
                setattr(self, '{}_span_pair_neg_label'.format(task_name),
                        self.vocab.get_token_index(BratDoc.NEG_SPAN_PAIR_LABEL, span_pair_label_ns))
                span_pair_num_classes = getattr(self, '{}_span_pair_num_classes'.format(task_name))
                span_pair_neg_label = getattr(self, '{}_span_pair_neg_label'.format(task_name))
                setattr(self, '{}_span_pair_layer'.format(task_name), span_pair_layer[task_name])
                setattr(self, '{}_span_pair_label_proj'.format(task_name),
                        SpanPairLabelProjectionLayer(span_pair_layer[task_name].get_output_dim(),
                                                     span_pair_num_classes,
                                                     span_pair_prediction_method))
                # metrics
                setattr(self, '{}_sp_acc'.format(task_name), MyCategoricalAccuracy(top_k=1, tie_break=False))
                setattr(self, '{}_sp_prf'.format(task_name), PrecisionRecallF1(neg_label=span_pair_neg_label))
                setattr(self, '{}_sp_prf_b'.format(task_name),
                        PrecisionRecallF1(neg_label=span_pair_neg_label, binary_match=True))

            # additional metrics
            if task_name not in self._special_metric:
                self._special_metric[task_name] = []
            for metric in self._special_metric[task_name]:
                if metric == 'coref':
                    setattr(self, '{}_coref'.format(task_name), MyConllCorefScores())
                elif metric == 'mr':  # mention recall
                    setattr(self, '{}_mr'.format(task_name), MentionRecall())
                elif metric == 'semeval_2010':  # micro f1 used in semeval 2010 task 8
                    setattr(self, '{}_semeval_2010'.format(task_name), Semeval2010(
                        vocab, '{}_span_pair_labels'.format(task_name), reduce='macro'))
                elif metric == 'bracket':
                    setattr(self, '{}_bracket'.format(task_name), EvalbBracketingScorer(DEFAULT_EVALB_DIR))
                elif metric == 'binary_sp_prf':
                    setattr(self, '{}_binary_sp_prf'.format(task_name),
                            PrecisionRecallF1(neg_label=span_pair_neg_label))
                else:
                    raise Exception('unsupported metrics')

        initializer(self)


    @overrides
    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                task_labels: torch.IntTensor,  # SHAPE: (batch_size)
                spans: torch.IntTensor,  # SHPAE: (batch_size, num_spans, 2)
                span_weights: torch.FloatTensor,  # SHAPE: (batch_size, num_spans)
                span_pairs: torch.IntTensor = None,  # SHPAE: (batch_size, num_span_pairs, 2)
                span_labels: torch.IntTensor = None,  # SHPAE: (batch_size, num_spans)
                span_pair_labels: torch.IntTensor = None,  # SHPAE: (batch_size, num_span_pairs)
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        '''
        Some tasks do not have span annotations or span-pair annotation.
        Since span_pair relies on span, span cannot be None but span_pair can.
        '''
        #time1 = time.time()

        task2e2e = dict([(m['task'], m['e2e']) for m in metadata])
        this_batch_task = metadata[0]['task']  # TODO: only allow one task per batch
        max_span_width = metadata[0]['max_span_width']

        # ------ shared layers ------
        # SHAPE: (batch_size, seq_len, emb_dim)
        if hasattr(self._text_field_embedder, 'token_embedder_bert') and \
                isinstance(self._text_field_embedder.token_embedder_bert, BertEmbedder):
            self._text_field_embedder.token_embedder_bert.set_task(this_batch_task)
        if self._mid_layer is not None:
            token_emb, mid_emb = self._text_field_embedder(text)
        else:
            token_emb = self._text_field_embedder(text)
        text_emb = self._lexical_dropout(token_emb)
        batch_size = text_emb.size(0)
        seq_len = text_emb.size(1)
        num_spans = spans.size(1)

        # SHAPE: (batch_size, seq_len)
        text_mask = util.get_text_field_mask(text).float()
        text_len = text_mask.sum(-1).long()

        # -1 is used for padding and some spans might be out of bounds, e.g., due to wordpiece tokenization
        # SHAPE: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).float()  # sanity check
        sent_len = text_mask.sum(-1, keepdim=True).long()
        span_mask_oos = ((spans[:, :, 0] >= sent_len) | (spans[:, :, 1] >= sent_len) | (spans[:, :, 1] - spans[:, :, 0] >= max_span_width)).float()  # spans not considered by the model
        span_mask = span_mask * (1 - span_mask_oos)

        spans = (spans.float() * span_mask.unsqueeze(-1)).long()

        if self._span_label_emb is not None:
            span_label_emb = self._span_label_emb(span_labels)
            text_emb = torch.cat([text_emb, span_label_emb], -1)

        if self._different_span_repr:
            context_layer = getattr(self, '_context_layer_{}'.format(this_batch_task))
        else:
            context_layer = self._context_layer

        if context_layer is not None:
            if self._different_span_repr:
                endpoint_span_extractor = getattr(self, '_endpoint_span_extractor_{}'.format(this_batch_task))
            else:
                endpoint_span_extractor = self._endpoint_span_extractor

        if self._different_span_repr:
            attentive_span_extractor = getattr(self, '_attentive_span_extractor_{}'.format(this_batch_task))
        else:
            attentive_span_extractor = self._attentive_span_extractor

        # collect span embedding
        span_emb_li = []
        if context_layer is not None:
            # SHAPE: (batch_size, seq_len, context_dim)
            cont_emb = context_layer(text_emb, text_mask)
            # SHAPE: (batch_size, num_spans, endpoint_dim)
            ep_span_emb = endpoint_span_extractor(cont_emb, spans)
            span_emb_li.append(ep_span_emb)
        if attentive_span_extractor is not None:
            if self._attentive_after_context:
                attn_inp = cont_emb
            else:
                attn_inp = text_emb
            if self._attentive_dim_reduction is not None:
                attn_inp = self._attentive_dim_reduction(attn_inp)
            # SHAPE: (batch_size, num_spans, emb_dim)
            att_span_emb = attentive_span_extractor(attn_inp, spans)
            att_span_emb = self._post_attentive(att_span_emb)
            span_emb_li.append(att_span_emb)
        span_emb = torch.cat(span_emb_li, -1)
        span_emb_dim = span_emb.size(-1)

        if self._log_sim_file:
            cos = pairwise_cosine_similarity(cont_emb)
            dump_similarity(cos, text_len, self._log_sim_file)

        if span_pairs is not None:
            num_span_pairs = span_pairs.size(1)

            # SHAPE: (batch_size)
            span_len = span_mask.sum(-1).long()

            # -1 is used as padding some spans might not exist anymore
            # SHAPE: (batch_size, num_span_pairs)
            span_pair_mask = ((span_pairs[:, :, 0] >= 0) &
                              (span_pairs[:, :, 0] < span_len.unsqueeze(-1)) &
                              (span_pairs[:, :, 1] < span_len.unsqueeze(-1))).float()
            span_pairs = (span_pairs.float() * span_pair_mask.unsqueeze(-1)).long()

        output_dict = {
            'loss': torch.tensor(0.0).to(text_emb.device),
            'task': {}
        }

        # ----- task-specific layers -----
        for task_ind, task_name in self.ind2task:
            task_mask = task_labels.eq(task_ind)
            if task_mask.sum().item() == 0:
                continue  # skip when the task is not found in the current batch

            task_loss = torch.tensor(0.0).to(text_emb.device)

            t_batch_size = task_mask.size(0)
            # SHAPE: (task_batch_size, seq_len)
            t_text_mask = text_mask.masked_select(task_mask.view(-1, 1)).view(-1, seq_len)
            output_dict['task'][task_name] = {}

            # ----- span -----
            # SHAPE: (task_batch_size, num_spans, 2)
            t_spans = spans.masked_select(task_mask.view(-1, 1, 1)).view(-1, num_spans, 2)
            # SHAPE: (task_batch_size, num_spans)
            t_span_len = t_spans[:, :, 1] - t_spans[:, :, 0] + 1
            # SHAPE: (task_batch_size, num_spans, span_emb_dim)
            t_span_emb = span_emb.masked_select(task_mask.view(-1, 1, 1)).view(
                -1, num_spans, span_emb_dim)
            if self._task_emb_dim:
                task_emb = self.task_emb(torch.tensor(task_ind).to(text_emb.device)).view(1, 1, -1)
                t_span_emb = torch.cat([t_span_emb, task_emb.expand(t_span_emb.size(0), t_span_emb.size(1), -1)], -1)
            # SHAPE: (task_batch_size, num_spans)
            t_span_mask = span_mask.masked_select(task_mask.view(-1, 1)).view(-1, num_spans)
            t_span_mask_oos = span_mask_oos.masked_select(task_mask.view(-1, 1)).view(-1, num_spans)
            # SHAPE: (task_batch_size, num_spans)
            t_span_weights = span_weights.masked_select(task_mask.view(-1, 1)).view(-1, num_spans)

            # select task-related layers
            span_layer = getattr(self, '{}_span_layer'.format(task_name))
            span_label_proj = getattr(self, '{}_span_label_proj'.format(task_name))

            # span neg label
            neg_label_ind = getattr(self, '{}_span_neg_label'.format(task_name))

            #time2 = time.time()

            # SHAPE: (task_batch_size, num_spans, num_classes)
            t_span_logits = span_label_proj(span_layer(t_span_emb))
            # SHAPE: (task_batch_size, num_spans, num_classes)
            t_span_prob = F.softmax(t_span_logits, dim=-1)
            # use mask to avoid invalid spans being selected by topk
            t_span_prob_masked = self.prob_mask(t_span_prob, t_span_mask, value=1.0)

            if task_name == 'coref' and self._special_loss[task_name]:
                t_span_logits = self.prob_mask(t_span_logits, t_span_mask, value=-1e20)
                mention_label = self.vocab.get_token_index('mention', 'coref_span_labels')
                t_span_neg_logit = t_span_logits[:, :, mention_label]  # mention score
            else:
                # SHAPE: (task_batch_size, num_spans)
                t_span_neg_logit = t_span_logits[:, :, neg_label_ind]

            # save to output
            output_dict['task'][task_name]['spans'] = t_spans
            output_dict['task'][task_name]['span_logits'] = t_span_logits
            output_dict['task'][task_name]['span_mask'] = t_span_mask
            output_dict['task'][task_name]['span_mask_oos'] = t_span_mask_oos
            output_dict['task'][task_name]['text_mask'] = t_text_mask
            if span_labels is not None:
                # SHAPE: (task_batch_size, num_spans)
                t_span_labels = span_labels.masked_select(task_mask.view(-1, 1)).view(-1, num_spans)
                output_dict['task'][task_name]['span_labels'] = t_span_labels

            # span loss
            if span_labels is not None and 'span' in self._task_loss[task_name]:
                if self.training or self._mid_layer is not None:
                    # use the most confusing negative spans (instead of all) for training
                    # and (optionally) refine their span representations
                    # TODO: sequences in the same batch keeps the same number of
                    #   negative spans despite their different length
                    num_spans_to_keep = self.get_num_spans_to_keep(task_name, seq_len, t_span_prob.size(1))
                    # the threshold to select top spans
                    # SHAPE: (task_batch_size, 1)
                    top_v = (-t_span_prob_masked[:, :, neg_label_ind]).topk(num_spans_to_keep, -1)[0][:, -1:]
                    # mask used to select top spans
                    # SHAPE: (task_batch_size, num_spans)
                    top_mask = t_span_prob[:, :, neg_label_ind] <= -top_v
                    # keep all positive samples and the other neg samples in top_mask for loss
                    t_span_mask_subset = t_span_mask * (top_mask | t_span_labels.ne(neg_label_ind)).float()

                    # TODO: how to set this value
                    # avoid top spans for span representation
                    num_spans_to_keep_half = max(int(num_spans_to_keep / 3), 1)
                    top_v_half = (-t_span_prob_masked[:, :, neg_label_ind]).topk(num_spans_to_keep_half, -1)[0][:, -1:]
                    top_mask_half = t_span_prob[:, :, neg_label_ind] <= -top_v_half
                    t_span_top = t_span_mask * top_mask_half.float()

                    if self._mid_layer is not None:
                        # pad t_span_top so that each sample in the mask has the same number of ones
                        # SHAPE: (batch_size,)
                        t_span_top_len = t_span_top.sum(-1).long()
                        max_t_span_top_len = t_span_top_len.max().item()
                        add_len = max_t_span_top_len - t_span_top_len  # number of ones to be added
                        add_mask = (torch.arange(max_t_span_top_len).to(t_span_top_len.device).view(1, -1)
                                    < add_len.view(-1, 1)).float()  # mask to be appended
                        # SHAPE: (batch_size, num_spans + max_t_span_top_len, 1)
                        add_mask = torch.cat([t_span_top, add_mask], -1).eq(1).unsqueeze(-1)

                        # extract span indices using the mask produced before
                        # SHAPE: (batch_size, num_spans + max_t_span_top_len, 2)
                        spans_padded = F.pad(spans, (0, 0, 0, max_t_span_top_len), mode='constant', value=0)
                        # SHAPE: (batch_size, max_t_span_top_len, 2)
                        t_spans_selected = spans_padded.masked_select(add_mask).view(-1, max_t_span_top_len, 2)

                        # represent spans selected
                        # SHAPE: (batch_size, max_t_span_top_len, emb_dim)
                        mid_emb_span = self._bert_span_layer(mid_emb, t_spans_selected)
                        ep_span_emb = endpoint_span_extractor(cont_emb, t_spans_selected)
                        mid_emb_span = torch.cat([ep_span_emb, mid_emb_span], -1)

                        # combine old representation with new one
                        new_t_span_emb = torch.zeros((
                            t_span_emb.size(0),
                            t_span_emb.size(1) + max_t_span_top_len,
                            t_span_emb.size(2))).to(t_span_emb.device)
                        new_t_span_emb.masked_scatter_(add_mask, mid_emb_span)
                        # SHAPE: (task_batch_size, num_spans, num_classes)
                        new_t_span_emb = new_t_span_emb[:, :t_span_emb.size(1), :]
                        add_mask = add_mask[:, :t_span_logits.size(1), :]
                        t_span_emb = t_span_emb * (1 - add_mask.float() * 0.5) + new_t_span_emb * 0.5  # residual addition

                        '''
                        # compute logits based on span representations
                        # SHAPE: (batch_size, max_t_span_top_len, num_classes)
                        selected_t_span_logits = span_label_proj(span_layer(mid_emb_span))

                        # combine old logits with new logits
                        # SHAPE: (task_batch_size, num_spans + max_t_span_top_len, num_classes)
                        new_t_span_logits = torch.zeros((
                            t_span_logits.size(0),
                            t_span_logits.size(1) + max_t_span_top_len,
                            t_span_logits.size(2))).to(t_span_logits.device)
                        new_t_span_logits.masked_scatter_(add_mask, selected_t_span_logits)
                        # SHAPE: (task_batch_size, num_spans, num_classes)
                        new_t_span_logits = new_t_span_logits[:, :t_span_logits.size(1), :]
                        add_mask = add_mask[:, :t_span_logits.size(1), :]
                        #t_span_logits = t_span_logits * (1 - add_mask.float()) + new_t_span_logits
                        t_span_logits = t_span_logits * (1 - add_mask.float() * 0.5) + new_t_span_logits * 0.5 # residual addition
                        '''

                        # compute new logits
                        t_span_logits = span_label_proj(span_layer(t_span_emb))

                        # update span probability
                        # SHAPE: (task_batch_size, num_spans, num_classes)
                        t_span_prob = F.softmax(t_span_logits, dim=-1)
                        # use mask to avoid invalid spans being selected by topk
                        t_span_prob_masked = self.prob_mask(t_span_prob, t_span_mask, value=1.0)

                # TODO: really need to train on all spans?
                if not self.training or not self._truncate_span_loss[task_name]:
                    # use all spans provided for testing
                    t_span_mask_subset = t_span_mask
                '''
                # use all spans provided for testing
                t_span_mask_subset = t_span_mask
                '''

                # multiple 01 mask with weights for weighted span loss
                span_loss = sequence_cross_entropy_with_logits(
                    t_span_logits, t_span_labels, t_span_mask_subset * t_span_weights,
                    average=self._task_loss_reduction[task_name])
                span_loss = span_loss * self._task_weight[task_name]
                if self._use_uncertainty_weight:
                    uw = getattr(self, '{}_uncertain_weight'.format(task_name))
                    span_loss = torch.exp(-uw) * span_loss + 0.5 * uw
                task_loss += span_loss

                # metrics
                getattr(self, '{}_s_acc'.format(task_name))(t_span_logits, t_span_labels, t_span_mask_subset)
                getattr(self, '{}_s_prf'.format(task_name))(
                    t_span_logits.max(-1)[1], t_span_labels, t_span_mask_subset.long(),
                    bucket_value=t_span_len, sig_test=False, mask_oos=t_span_mask_oos.long())
                getattr(self, '{}_s_prf_b'.format(task_name))(
                    t_span_logits.max(-1)[1], t_span_labels, t_span_mask_subset.long())
                for special_metric in self._special_metric[task_name]:
                    if special_metric == 'bracket' and not self.training:
                        batch_gold_trees = [m.get('tree') for m in metadata]
                        raw_tokens = [m.get('raw_tokens') for m in metadata]
                        if all(batch_gold_trees):
                            gold_pos_tags: List[List[str]] = [list(zip(*tree.pos()))[1] for tree in batch_gold_trees]
                            predicted_trees = construct_trees(
                                self.vocab, '{}_span_labels'.format(task_name),
                                t_span_prob.cpu().data, t_spans.cpu().data, t_span_mask.sum(-1).long().data,
                                raw_tokens, gold_pos_tags)
                            getattr(self, '{}_bracket'.format(task_name))(predicted_trees, batch_gold_trees)

            # ----- span pair -----
            if span_pairs is None or 'span_pair' not in self._task_loss[task_name]:
                # track loss
                output_dict['loss'] += task_loss
                self.multi_loss(task_name, task_loss.item(), count=1)
                continue

            # SHAPE: (task_batch_size, num_span_pairs, 2)
            t_span_pairs = span_pairs.masked_select(task_mask.view(-1, 1, 1)).view(-1, num_span_pairs, 2)
            # SHAPE: (task_batch_size, num_span_pairs)
            t_span_pair_mask = span_pair_mask.masked_select(task_mask.view(-1, 1)).view(-1, num_span_pairs)

            span_pair_layer = getattr(self, '{}_span_pair_layer'.format(task_name))
            span_pair_label_proj = getattr(self, '{}_span_pair_label_proj'.format(task_name))

            if not task2e2e[task_name]:
                # use provided span pairs to construct embedding

                # get span pair embedding
                # SHAPE: (batch_size * num_span_pairs * 2)
                flat_span_pairs = util.flatten_and_batch_shift_indices(t_span_pairs, num_spans)

                # span pair prediction
                # SHAPE: (batch_size, num_span_pairs, num_classes)
                t_span_pair_logits = span_pair_label_proj(span_pair_layer(t_span_emb, t_span_pairs))

                # get negative span logits of the pair by sum
                # SHAPE: (batch_size, num_span_pairs, 2)
                t_span_pair_neg_logit = util.batched_index_select(
                    t_span_neg_logit.unsqueeze(-1), t_span_pairs, flat_span_pairs).squeeze(-1)
                # SHAPE: (batch_size, num_span_pairs)
                t_span_pair_neg_logit = t_span_pair_neg_logit.sum(-1)
                # SHAPE: (batch_size, num_span_pairs, 2)
                t_span_pair_len = util.batched_index_select(
                    t_span_len.unsqueeze(-1), t_span_pairs, flat_span_pairs).squeeze(-1)
                # SHAPE: (batch_size, num_span_pairs)
                t_span_pair_len = t_span_pair_len.max(-1)[0]

            else:
                # select span pairs by span scores to construct embedding

                ref_t_span_pairs = t_span_pairs
                ref_t_span_pair_mask = t_span_pair_mask

                # rank spans
                # TODO: sequences in the same batch keeps the same number of spans despite their different length
                num_spans_to_keep = self.get_num_spans_to_keep(task_name, seq_len, t_span_prob.size(1))
                if task_name == 'coref' and self._special_loss[task_name]:
                    mention_label = self.vocab.get_token_index('mention', 'coref_span_labels')
                    _, top_ind = t_span_logits[:, :, mention_label].topk(num_spans_to_keep, -1)
                else:
                    # SHAPE: (task_batch_size, num_spans_to_keep)
                    _, top_ind = (-t_span_prob_masked[:, :, neg_label_ind]).topk(num_spans_to_keep, -1)
                # sort according to the order (not strictly based on order because spans overlap)
                top_ind = top_ind.sort(-1)[0]

                # get out-of-bound mask
                # TODO: span must be located at the beginning
                # SHAPE: (task_batch_size, num_spans_to_keep)
                top_ind_mask = top_ind < t_span_mask.sum(-1, keepdim=True).long()

                # get pairs
                num_spans_to_keep = top_ind.size(1)
                external2internal = self.extenral_to_internal(top_ind, num_spans)

                if task_name == 'orl' and self._pair_ind_method[task_name] == 'gold_predicate':
                    # get gold predicate
                    pred_label = self.vocab.get_token_index('sentiment', 'orl_span_labels')
                    # SHAPE: (batch_size, num_spans)
                    pred_mask = t_span_labels.eq(pred_label).long() * t_span_mask.long()
                    num_pred = pred_mask.sum(-1)
                    max_num_pred = int(num_pred.max().item())
                    # SHAPE: (batch_size, max_num_pred)
                    _, pred_ind = pred_mask.topk(max_num_pred, -1)
                    # SHAPE: (batch_size, max_num_pred)
                    pred_ind_mask = torch.arange(max_num_pred, device=pred_ind.device).unsqueeze(0)
                    pred_ind_mask = pred_ind_mask < num_pred.unsqueeze(-1)

                    # SHAPE: (batch_size * num_span_pairs * 2)
                    t_span_pairs, t_span_pair_mask, t_span_pair_shape = self.span_ind_to_pair_ind(
                        top_ind, top_ind_mask, start_span_ind=pred_ind, start_span_ind_mask=pred_ind_mask,
                        method=self._pair_ind_method[task_name], absolute=False)
                else:
                    # SHAPE: (batch_size * num_span_pairs * 2)
                    t_span_pairs, t_span_pair_mask, t_span_pair_shape = self.span_ind_to_pair_ind(
                        top_ind, top_ind_mask, method=self._pair_ind_method[task_name], absolute=False)

                t_span_pairs_internal = external2internal(t_span_pairs)

                # get negative span logits of the pair by sum
                # SHAPE: (batch_size * num_span_pairs * 2)
                flat_span_pairs = util.flatten_and_batch_shift_indices(t_span_pairs, num_spans)
                # SHAPE: (batch_size, num_span_pairs, 2)
                t_span_pair_neg_logit = util.batched_index_select(
                    t_span_neg_logit.unsqueeze(-1), t_span_pairs, flat_span_pairs).squeeze(-1)
                # SHAPE: (batch_size, num_span_pairs)
                t_span_pair_neg_logit = t_span_pair_neg_logit.sum(-1)
                # SHAPE: (batch_size, num_span_pairs, 2)
                t_span_pair_len = util.batched_index_select(
                    t_span_len.unsqueeze(-1), t_span_pairs, flat_span_pairs).squeeze(-1)
                # SHAPE: (batch_size, num_span_pairs)
                t_span_pair_len = t_span_pair_len.max(-1)[0]

                # get span kept
                # SHAPE: (batch_size * num_spans_to_keep)
                flat_top_ind = util.flatten_and_batch_shift_indices(top_ind, num_spans)
                # SHAPE: (batch_size, num_spans_to_keep, 2)
                t_spans_for_pair = util.batched_index_select(t_spans, top_ind, flat_top_ind)
                # SHAPE: (batch_size, num_spans_to_keep, span_emb_dim)
                t_span_emb_for_pair = util.batched_index_select(t_span_emb, top_ind, flat_top_ind)
                # SHAPE: (batch_size, num_spans_to_keep, num_classe)
                t_span_prob_for_pair = util.batched_index_select(t_span_prob, top_ind, flat_top_ind)

                # high order propagation
                if self._num_order:
                    ns1, ns2 = t_span_pair_shape
                    high_order_layer = getattr(self, '{}_high_order_layer'.format(task_name))
                    t_span_emb_for_pair = high_order_layer(
                        t_span_emb_for_pair,
                        t_span_pairs_internal.view(-1, ns1, ns2, 2),
                        t_span_pair_mask.view(-1, ns1, ns2),
                        span_pair_layer,
                        span_pair_label_proj,
                        task_name)

                # span pair prediction
                # SHAPE: (batch_size, num_span_pairs, num_classes)
                t_span_pair_logits = span_pair_label_proj(span_pair_layer(t_span_emb_for_pair, t_span_pairs_internal))

            # span pair neg label
            neg_pair_label_ind = getattr(self, '{}_span_pair_neg_label'.format(task_name))

            if self._combine_span_and_pair_score[task_name]:
                if task_name == 'coref' and self._special_loss[task_name]:
                    coref_label = self.vocab.get_token_index('coreference', 'coref_span_pair_labels')
                    t_span_pair_logits[:, :, coref_label] += t_span_pair_neg_logit
                else:
                    t_span_pair_logits[:, :, neg_pair_label_ind] += t_span_pair_neg_logit

            t_span_pair_prob = F.softmax(t_span_pair_logits, dim=-1)

            # span pair loss
            if span_pair_labels is not None:
                t_span_pair_mask_for_loss = t_span_pair_mask

                if not task2e2e[task_name]:
                    # SHAPE: (task_batch_size, num_span_pairs)
                    t_span_pair_labels = span_pair_labels.masked_select(
                        task_mask.view(-1, 1)).view(-1, num_span_pairs)
                else:
                    # SHAPE: (task_batch_size, num_span_pairs)
                    ref_t_span_pair_labels = span_pair_labels.masked_select(
                        task_mask.view(-1, 1)).view(-1, num_span_pairs)
                    # SHAPE: (task_batch_size, num_spans_to_keep * num_spans_to_keep)
                    t_span_pair_labels = self.label_span_pair(
                        task_name, t_span_pairs, ref_t_span_pairs, ref_t_span_pair_labels, ref_t_span_pair_mask)

                    if task_name == 'orl':
                        # SHAPE: (task_batch_size, num_spans_to_keep * num_spans_to_keep)
                        t_span_pair_labels_binary = self.label_span_pair(
                            task_name, t_span_pairs, ref_t_span_pairs, ref_t_span_pair_labels, ref_t_span_pair_mask,
                            spans=t_spans, use_binary=True, span_pair_pred=t_span_pair_prob.max(-1)[1])

                    if task_name == 'srl' and self._special_loss[task_name]:
                        t_span_pair_mask_for_loss = self.span_labels_to_eval_mask(
                            t_span_prob_for_pair,
                            t_span_pair_mask.view(t_batch_size, num_spans_to_keep, num_spans_to_keep),
                            task_name='srl', parent_label='Predicate', child_label='Argument',
                            parent_ratio=0.4, child_ratio=0.8)
                        t_span_pair_mask_for_loss = t_span_pair_mask_for_loss.view(t_batch_size, -1)
                    elif task_name == 'coref' and self._special_loss[task_name]:
                        num_spans_to_keep1, num_spans_to_keep2 = t_span_pair_shape
                        # mask out neg pair label and padding label
                        pad_label = self.vocab.get_token_index(BratReader.PADDING_LABEL, 'coref_span_pair_labels')
                        t_span_pair_logits[:, :, pad_label] = -1.0
                        t_span_pair_logits[:, :, neg_pair_label_ind] = 0.0
                        t_span_pair_prob = F.softmax(t_span_pair_logits, dim=-1)
                        span_pair_loss = self.all_vs_one_loss(
                            t_span_pair_logits.view(t_batch_size, num_spans_to_keep1, num_spans_to_keep2, -1),
                            t_span_pair_mask.view(t_batch_size, num_spans_to_keep1, num_spans_to_keep2),
                            t_span_pair_labels.view(t_batch_size, num_spans_to_keep1, num_spans_to_keep2),
                            task_name=task_name,
                            major_label='coreference',
                            add_dummy=True)

                if task_name == 'coref' and self._special_loss[task_name]:
                    pass
                else:
                    span_pair_loss = sequence_cross_entropy_with_logits(
                        t_span_pair_logits, t_span_pair_labels, t_span_pair_mask_for_loss,
                        average=self._task_loss_reduction[task_name])

                span_pair_loss = span_pair_loss * self._task_weight[task_name]
                if self._use_uncertainty_weight:
                    uw = getattr(self, '{}_uncertain_weight'.format(task_name))
                    span_pair_loss = torch.exp(-uw) * span_pair_loss + 0.5 * uw
                task_loss += span_pair_loss

                # metrics
                getattr(self, '{}_sp_acc'.format(task_name))(
                    t_span_pair_logits, t_span_pair_labels, t_span_pair_mask)
                if not task2e2e[task_name]:
                    recall = None
                else:
                    recall = ref_t_span_pair_labels.ne(neg_pair_label_ind)
                    recall = (recall.float() * ref_t_span_pair_mask).long()

                if task_name == 'dp' and task2e2e[task_name]:  # TODO: add config
                    t_span_pair_mask, t_span_pair_prob = self.span_pairs_to_eval_mask(
                        t_span_pairs.view(t_batch_size, num_spans_to_keep, num_spans_to_keep, 2),
                        t_span_pair_prob.view(t_batch_size, num_spans_to_keep, num_spans_to_keep, -1),
                        t_span_pair_mask.view(t_batch_size, num_spans_to_keep, num_spans_to_keep),
                        task_name=task_name,
                        direction='no_self',
                        skip_first_span=True,
                        only_one_inlink=True)
                    t_span_pair_mask = t_span_pair_mask.view(t_batch_size, -1)
                    t_span_pair_prob = t_span_pair_prob.view(t_batch_size, num_spans_to_keep * num_spans_to_keep, -1)

                # SHAPE: (task_batch_size, num_span_pairs)
                t_span_pair_pred = t_span_pair_prob.max(-1)[1]

                # save to output
                output_dict['task'][task_name]['span_pairs'] = t_span_pairs
                output_dict['task'][task_name]['span_pair_preds'] = t_span_pair_pred
                output_dict['task'][task_name]['span_pair_mask'] = t_span_pair_mask.long()
                output_dict['task'][task_name]['span_pair_labels'] = t_span_pair_labels

                if task2e2e[task_name]:
                    output_dict['task'][task_name]['ref_span_pairs'] = ref_t_span_pairs
                    output_dict['task'][task_name]['ref_span_pair_mask'] = ref_t_span_pair_mask.long()
                    output_dict['task'][task_name]['ref_span_pair_labels'] = ref_t_span_pair_labels

                getattr(self, '{}_sp_prf'.format(task_name))(
                    t_span_pair_pred, t_span_pair_labels, t_span_pair_mask.long(),
                    recall=recall, bucket_value=t_span_pair_len, sig_test=False)
                getattr(self, '{}_sp_prf_b'.format(task_name))(
                    t_span_pair_pred, t_span_pair_labels, t_span_pair_mask.long(), recall=recall)
                for special_metric in self._special_metric[task_name]:
                    if special_metric == 'coref':
                        coref_label = self.vocab.get_token_index(
                            'coreference', '{}_span_pair_labels'.format(task_name))
                        # SHAPE: (task_batch_size, num_spans_to_keep)
                        coref_parent = self.get_parent(
                            t_span_pair_prob.view((t_batch_size,) + t_span_pair_shape + (-1,)),
                            t_span_pair_mask.view((t_batch_size,) + t_span_pair_shape),
                            coref_label)
                        getattr(self, '{}_coref'.format(task_name))(
                            t_spans_for_pair,
                            t_span_pairs_internal.view((t_batch_size,) + t_span_pair_shape + (2,))[:, :, :, 1],
                            coref_parent, metadata)
                    elif special_metric == 'mr':
                        # TODO: add mask?
                        getattr(self, '{}_mr'.format(task_name))(t_spans_for_pair, metadata)
                    elif special_metric == 'semeval_2010':
                        getattr(self, '{}_semeval_2010'.format(task_name))(
                            t_span_pair_pred, t_span_pair_labels, t_span_pair_mask.long())
                    elif special_metric == 'binary_sp_prf':
                        getattr(self, '{}_binary_sp_prf'.format(task_name))(
                            t_span_pair_pred, t_span_pair_labels_binary, t_span_pair_mask.long(),
                            recall=recall, duplicate_check=False)

            # track loss
            output_dict['loss'] += task_loss
            self.multi_loss(task_name, task_loss.item(), count=1)

        #time3 = time.time()
        #print('\n{}\t{}\t{}\n'.format('TIME', time2 - time1, time3 - time2))

        return output_dict


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for task_name, task_output in output_dict['task'].items():
            output_dict['span_with_label'] = []
            if 'span_logits' in task_output and 'span_mask' in task_output:
                # SHAPE: (batch_size, num_spans, 2)
                spans = task_output['spans'].cpu().numpy()
                # SHAPE: (batch_size, num_spans, num_class)
                logits = task_output['span_logits']
                # SHAPE: (batch_size, num_spans)
                preds = logits.max(-1)[1].cpu().numpy()
                # SHAPE: (batch_size, num_spans)
                mask = task_output['span_mask'].cpu().numpy()
                mask_oos = task_output['span_mask_oos'].cpu().numpy()
                # SHAPE: (batch_size, num_spans)
                labels = task_output['span_labels'].cpu().numpy()
                # SHAPE: (batch_size, seq_len)
                text_mask = task_output['text_mask'].cpu().numpy()
                bs, ns = mask.shape

                neg_label_ind = getattr(self, '{}_span_neg_label'.format(task_name))
                namespace = '{}_span_labels'.format(task_name)
                ind2label = lambda ind: self.vocab.get_token_from_index(ind, namespace)

                for b in range(bs):
                    sl: List[Tuple[Tuple, str, str]] = []
                    output_dict['span_with_label'].append(sl)
                    for s in range(ns):
                        if mask[b, s] == 0 and mask_oos[b, s] == 0:
                            continue
                        span_boundary: Tuple[int, int] = tuple(spans[b, s])
                        p_label: int = neg_label_ind if mask_oos[b, s] else preds[b, s]
                        g_label: int = labels[b, s]
                        if p_label == neg_label_ind and g_label == neg_label_ind:
                            continue
                        p_label: str = ind2label(p_label)
                        g_label: str = ind2label(g_label)
                        sl.append((span_boundary, p_label, g_label))

            output_dict['span_pair_with_label'] = []
            if 'span_pair_preds' in task_output and 'span_pair_mask' in task_output:
                # SHAPE: (batch_size, num_span_pairs, 2)
                span_pairs = task_output['span_pairs'].cpu().numpy()
                # SHAPE: (batch_size, num_span_pairs)
                span_pair_preds = task_output['span_pair_preds'].cpu().numpy()
                # SHAPE: (batch_size, num_spans)
                span_pair_mask = task_output['span_pair_mask'].cpu().numpy()
                # SHAPE: (batch_size, num_spans)
                span_pair_labels = task_output['span_pair_labels'].cpu().numpy()
                bs, nsp = span_pair_mask.shape

                if 'ref_span_pairs' in task_output:
                    # SHAPE: (batch_size, num_span_pairs, 2)
                    ref_span_pairs = task_output['ref_span_pairs'].cpu().numpy()
                    # SHAPE: (batch_size, num_spans)
                    ref_span_pair_mask = task_output['ref_span_pair_mask'].cpu().numpy()
                    # SHAPE: (batch_size, num_spans)
                    ref_span_pair_labels = task_output['ref_span_pair_labels'].cpu().numpy()
                    _, ref_nsp = ref_span_pair_mask.shape

                neg_sp_label_ind = getattr(self, '{}_span_pair_neg_label'.format(task_name))
                sp_namespace = '{}_span_pair_labels'.format(task_name)
                sp_ind2label = lambda ind: self.vocab.get_token_from_index(ind, sp_namespace)

                for b in range(bs):
                    spl: List[Tuple[Tuple, Tuple, str, str]] = []
                    output_dict['span_pair_with_label'].append(spl)
                    added: Set[Tuple[int, int]] = set()
                    for s in range(nsp):
                        if span_pair_mask[b, s] == 0:
                            continue
                        added.add((span_pairs[b, s, 0], span_pairs[b, s, 1]))
                        s1b: Tuple[int, int] = tuple(spans[b, span_pairs[b, s, 0]])
                        s2b: Tuple[int, int] = tuple(spans[b, span_pairs[b, s, 1]])
                        p_label: int = span_pair_preds[b, s]
                        g_label: int = span_pair_labels[b, s]
                        if p_label == neg_sp_label_ind and g_label == neg_sp_label_ind:
                            continue
                        p_label: str = sp_ind2label(p_label)
                        g_label: str = sp_ind2label(g_label)
                        spl.append((s1b, s2b, p_label, g_label))

                    if 'ref_span_pairs' in task_output:
                        for s in range(ref_nsp):
                            if ref_span_pair_mask[b, s] == 0 or (ref_span_pairs[b, s, 0], ref_span_pairs[b, s, 1]) in added:
                                continue
                            s1b: Tuple[int, int] = tuple(spans[b, ref_span_pairs[b, s, 0]])
                            s2b: Tuple[int, int] = tuple(spans[b, ref_span_pairs[b, s, 1]])
                            p_label: int = neg_sp_label_ind
                            g_label: int = ref_span_pair_labels[b, s]
                            if p_label == neg_sp_label_ind and g_label == neg_sp_label_ind:
                                continue
                            p_label: str = sp_ind2label(p_label)
                            g_label: str = sp_ind2label(g_label)
                            spl.append((s1b, s2b, p_label, g_label))

        return output_dict


    def decode_(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # TODO: a better way to organize output
        #   currently the order in the output might not be the same as input
        #   when there are multiple tasks
        output_dict['pred_bio'] = []
        output_dict['gold_bio'] = []

        for task_name, task_output in output_dict['task'].items():
            if 'span_logits' in task_output and 'span_mask' in task_output:
                # SHAPE: (batch_size, num_spans, 2)
                spans = task_output['spans'].cpu().numpy()
                # SHAPE: (batch_size, num_spans)
                logits = task_output['span_logits']
                preds = logits.max(-1)[1].cpu().numpy()
                # SHAPE: (batch_size, num_spans)
                mask = task_output['span_mask'].cpu().numpy()
                # SHAPE: (batch_size, num_spans)
                labels = task_output['span_labels'].cpu().numpy()
                # SHAPE: (batch_size, seq_len)
                text_mask = task_output['text_mask'].cpu().numpy()
                bs, ns = mask.shape

                neg_label_ind = getattr(self, '{}_span_neg_label'.format(task_name))
                namespace = '{}_span_labels'.format(task_name)
                ind2label = lambda ind: self.vocab.get_token_from_index(ind, namespace)

                task_output['pred_bio'] = []
                task_output['gold_bio'] = []
                for b in range(bs):
                    sent_len = int(text_mask[b].sum().item())
                    pred_spans: Dict[Tuple[int, int], str] = {}
                    gold_spans: Dict[Tuple[int, int], str] = {}
                    for s in range(ns):
                        if mask[b, s] == 0:
                            continue
                        if preds[b, s] != neg_label_ind:
                            pred_spans[tuple(spans[b, s])] = ind2label(preds[b, s])
                        if labels[b, s] != neg_label_ind:
                            gold_spans[tuple(spans[b, s])] = ind2label(labels[b, s])

                    # generate BIO tags
                    pred_bio = self.spans_to_bio(pred_spans, sent_len)
                    gold_bio = self.spans_to_bio(gold_spans, sent_len)
                    output_dict['pred_bio'].append(pred_bio)
                    output_dict['gold_bio'].append(gold_bio)

        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        def update_metric(m, result):
            if not hasattr(self, m):
                return
            v = getattr(self, m).get_metric(reset=reset)
            if v is None:
                return
            elif type(v) is float:
                result[m] = v
            elif type(v) is dict:
                for k in v:
                    result[m + '_' + k] = v[k]
            elif type(v) is tuple:
                for i, k in enumerate(v):
                    result[m + '_' + str(i)] = k
            else:
                raise NotImplementedError
        result = {}
        for task_ind, task_name in self.ind2task:
            for loss in ['s', 'sp']:
                for metric in ['acc', 'prf', 'prf_b']:
                    m = '{}_{}_{}'.format(task_name, loss, metric)
                    update_metric(m, result)
            for metric in self._special_metric[task_name]:
                m = '{}_{}'.format(task_name, metric)
                update_metric(m, result)
        result.update(self.multi_loss.get_metric(reset=reset))
        return result


    def has_overlap(self, span1: Tuple[int, int], span2: Tuple[int, int]):
        return not (span1[0] > span2[1] or span2[0] > span1[1])


    def label_span_pair(self,
                        task_name: str,
                        span_pairs: torch.IntTensor,  # SHAPE: (batch_size, num_span_pairs1, 2)
                        ref_span_pairs: torch.IntTensor,  # SHAPE: (batch_size, num_span_pairs2, 2)
                        ref_span_pair_labels: torch.IntTensor,  # SHPAE: (batch_size, num_span_pairs2)
                        ref_span_pair_mask: torch.FloatTensor,  # SHAPE: (batch_size, num_span_pairs2)
                        spans: torch.IntTensor = None,  # SHAPE: (batch_size, num_spans, 2)
                        use_binary: bool = False,
                        span_pair_pred: torch.IntTensor = None  # SHAPE: (batch_size, num_span_pairs1)
                        ) -> torch.IntTensor: # SHPAE: (batch_size, num_span_pairs1)
        neg_label_ind = getattr(self, '{}_span_pair_neg_label'.format(task_name))
        device = span_pairs.device
        span_pairs = span_pairs.cpu().numpy()
        ref_span_pairs = ref_span_pairs.cpu().numpy()
        ref_span_pair_labels = ref_span_pair_labels.cpu().numpy()
        ref_span_pair_mask = ref_span_pair_mask.cpu().numpy()
        batch_size = ref_span_pairs.shape[0]
        ref_num_span_pairs = ref_span_pairs.shape[1]
        num_span_pairs = span_pairs.shape[1]

        if spans is not None and use_binary:
            spans = spans.cpu().numpy()
            if span_pair_pred is not None:
                span_pair_pred = span_pair_pred.cpu().numpy()

        span_pair_labels = []
        for b in range(batch_size):
            label_dict = defaultdict(lambda: neg_label_ind)
            label_dict.update(dict((tuple(ref_span_pairs[b, i]), ref_span_pair_labels[b, i])
                                   for i in range(ref_num_span_pairs) if ref_span_pair_mask[b, i] > 0))
            labels = []
            for i in range(num_span_pairs):
                tsp1, tsp2 = tuple(span_pairs[b, i])
                assign_label = label_dict[(tsp1, tsp2)]
                if span_pair_pred is not None:
                    pred_label = span_pair_pred[b, i]
                else:
                    pred_label = None
                if pred_label == neg_label_ind:  # skip pairs not predicated as positive
                    labels.append(assign_label)
                    continue
                if spans is not None and use_binary:
                    # find overlapping span pairs
                    has_overlap = False
                    for (sp1, sp2), l in label_dict.items():
                        if l == neg_label_ind:
                            continue
                        if pred_label and l != pred_label:  # only look at ground truth with predicted label
                            continue
                        if self.has_overlap(spans[b, tsp1], spans[b, sp1]) and \
                                self.has_overlap(spans[b, tsp2], spans[b, sp2]):
                            assign_label = l
                            has_overlap = True
                labels.append(assign_label)
            span_pair_labels.append(labels)
        return torch.LongTensor(span_pair_labels).to(device)


    def prob_mask(self,
                  prob: torch.FloatTensor,
                  mask: torch.FloatTensor,
                  value: float = 1.0):
        ''' Add value to the positions masked out. prob is larger than mask by one dim. '''
        return prob + ((1.0 - mask) * value).unsqueeze(-1)


    def spans_to_bio(self, spans: Dict[Tuple[int, int], str], sent_len: int) -> List[str]:
        bio = np.full(sent_len, 'O', dtype=object)
        touched = np.full(sent_len, False)
        for (start, end), label in spans.items():
            if np.any(touched[start:end + 1]):  # skip if there is overlap
                continue
            bio[start] = 'B-' + label
            bio[start + 1:end + 1] = 'I-' + label
        return bio


    def span_labels_to_eval_mask(self,
                                 # SHAPE: (batch_size, num_spans, num_class)
                                 span_prob: torch.LongTensor,
                                 # SHAPE: (batch_size, num_spans, num_spans)
                                 span_pair_mask: torch.FloatTensor,
                                 task_name: str,
                                 parent_label: str,
                                 child_label: str,
                                 parent_ratio: float,
                                 child_ratio: float):
        batch_size, num_spans, _ = span_prob.size()
        parent_label = self.vocab.get_token_index(parent_label, '{}_span_labels'.format(task_name))
        child_label = self.vocab.get_token_index(child_label, '{}_span_labels'.format(task_name))
        parent_prob = span_prob[:, :, parent_label]
        child_prob = span_prob[:, :, child_label]
        parent_ratio = max(int(num_spans * parent_ratio), 1)
        child_ratio = max(int(num_spans * child_ratio), 1)

        # SHAPE: (batch_size, parent_ratio)
        _, parent_ind = parent_prob.topk(parent_ratio, -1)
        # SHAPE: (batch_size, child_ratio)
        _, child_ind = child_prob.topk(child_ratio, -1)

        parent_mask = torch.zeros((batch_size, num_spans), dtype=torch.uint8, device=span_pair_mask.device)
        parent_mask.scatter_(1, parent_ind, 1)
        child_mask = torch.zeros((batch_size, num_spans), dtype=torch.uint8, device=span_pair_mask.device)
        child_mask.scatter_(1, child_ind, 1)

        label_span_pair_mask = (parent_mask.unsqueeze(2) & child_mask.unsqueeze(1)).float()

        return span_pair_mask * label_span_pair_mask


    def span_pairs_to_eval_mask(self,
                                # SHAPE: (batch_size, num_spans, num_spans, 2)
                                span_pairs: torch.LongTensor,
                                # SHAPE: (batch_size, num_spans, num_spans, num_classes)
                                span_pair_prob: torch.FloatTensor,
                                # SHAPE: (batch_size, num_spans, num_spans)
                                span_pair_mask: torch.FloatTensor,
                                task_name: str,
                                direction: str = None,
                                # used for dependency parsing where the first span is usually root
                                skip_first_span: bool = False,
                                # used for dependency parsing where result is a tree
                                only_one_inlink: bool = False
                                # SHAPE: (batch_size, num_spans, num_spans),
                                #  (batch_size, num_spans, num_spans, num_classes)
                                ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
        Get the evaluation mask given span pairs (all pairs among num_spans spans).
        (1) Dependency parsing: we need to find exactly one head for each word (except for the first word "ROOT").
        (2) Correference resolution: all the edges should point to the previous words (TODO: not implemented)
        '''
        assert direction in {'left', 'right', 'no_self', None}
        # Assume that spans' indices are consistent with the order they appear in the sentence.
        # Otherwise we cannot infer the order of two spans given their indices.
        # If spans overlap, they might be ordered by start/end or end/start.
        # No matter which ordering is used, the orders among non-overlapping spans are the same.
        parent_spans, child_spans = span_pairs[:, :, :, 0], span_pairs[:, :, :, 1]
        if direction == 'left':  # get directional mask that only allows edges going left, e.g, coref
            # SHAPE: (batch_size, num_spans, num_spans)
            direction_mask = (parent_spans > child_spans).float()
        elif direction == 'right':  # get directional mask that only allows edges going right, e.g., craft coref
            # SHAPE: (batch_size, num_spans, num_spans)
            direction_mask = (parent_spans < child_spans).float()
        elif direction == 'no_self':  # get directional mask that forbids self-edges, e.g., dependency parsing
            # SHAPE: (batch_size, num_spans, num_spans)
            direction_mask = parent_spans.ne(child_spans).float()
        else:
            direction_mask = torch.ones_like(span_pair_mask)

        span_pair_mask = span_pair_mask * direction_mask

        if skip_first_span:
            span_pair_mask *= child_spans.ne(0).float()

        if only_one_inlink:
            # mask out disallowed pairs
            span_pair_prob *= span_pair_mask.unsqueeze(-1)
            # mask out negative span pair labels
            span_pair_prob_neg = torch.ones_like(span_pair_prob)
            neg_pair_label_ind = getattr(self, '{}_span_pair_neg_label'.format(task_name))
            span_pair_prob_neg[:, :, :, neg_pair_label_ind] = 0.0
            span_pair_prob *= span_pair_prob_neg

            num_spans = span_pairs.size(1)
            # for each span pair, find the label with max logit
            # SHAPE: (batch_size, num_spans, num_spans)
            span_pair_label_prob, span_pair_label_ind = span_pair_prob.max(-1)
            # for each child span, find the parent with max logit
            # SHAPE: (batch_size, num_spans)
            span_label_prob, span_arc_ind = span_pair_label_prob.max(1)  # mask out parent dimension

            # get the new mask the has only a single "1" along the parent dimension
            # SHAPE: (1, num_spans, 1)
            span_pair_single_mask = torch.arange(num_spans, device=span_pairs.device).unsqueeze(0).unsqueeze(-1)
            # SHAPE: (batch_size, num_spans, num_spans)
            span_pair_single_mask = span_pair_single_mask.eq(span_arc_ind.unsqueeze(1)).float()
            span_pair_mask *= span_pair_single_mask

        return span_pair_mask, span_pair_prob


    def span_ind_to_pair_ind(self,
                             span_ind: torch.LongTensor,  # SHAPE: (batch_size, num_spans)
                             span_ind_mask: torch.FloatTensor,  # SHAPE: (batch_size, num_spans)
                             start_span_ind: torch.LongTensor = None,  # SHAPE: (batch_size, num_spans2)
                             start_span_ind_mask: torch.FloatTensor = None,  # SHAPE: (batch_size, num_spans2)
                             method: str = None,
                             absolute: bool = True) -> Tuple[torch.LongTensor, torch.FloatTensor, Tuple]:
        ''' Create span pair indices and corresponding mask based on selected spans '''
        batch_size, num_spans = span_ind.size()

        if method and method.startswith('left:'):
            left_size = int(method.split(':', 1)[1])

            # get mask
            # span indices should be in the same order as they appear in the sentence
            if absolute:
                # SHAPE: (batch_size, num_spans, num_spans)
                left_mask = (span_ind.unsqueeze(1) < span_ind.unsqueeze(2)) & \
                            (span_ind.unsqueeze(1) >= (span_ind.unsqueeze(2) - left_size))
            else:
                # SHAPE: (num_spans,)
                end_boundary = torch.arange(num_spans, device=span_ind.device)
                start_boundary = end_boundary - left_size
                # SHAPE: (num_spans, num_spans)
                left_mask = (end_boundary.unsqueeze(0) < end_boundary.unsqueeze(-1)) & \
                            (end_boundary.unsqueeze(0) >= start_boundary.unsqueeze(-1))
                left_mask = left_mask.unsqueeze(0).repeat(batch_size, 1, 1)

            # SHAPE: (batch_size, num_spans)
            left_mask_num = left_mask.sum(-1)
            left_mask_num_max = max(left_mask_num.max().item(), 1)  # keep at least 1 span pairs to avoid bugs
            # SHAPE: (batch_size, num_spans)
            left_mask_num_left = left_mask_num_max - left_mask_num
            # SHAPE: (1, 1, left_mask_num_max)
            left_mask_ext = torch.arange(left_mask_num_max, device=span_ind.device).unsqueeze(0).unsqueeze(0)
            # SHAPE: (batch_size, num_spans, left_mask_num_max)
            left_mask_ext = left_mask_ext < left_mask_num_left.unsqueeze(-1)
            # SHAPE: (batch_size, num_spans, num_spans + left_mask_num_max)
            left_mask = torch.cat([left_mask, left_mask_ext], -1)

            # extend span_ind and span_ind_mask
            # SHAPE: (batch_size, num_spans + left_mask_num_max)
            span_ind_child = torch.cat([span_ind,
                                        span_ind.new_zeros((batch_size, left_mask_num_max))], -1)
            span_ind_child_mask = torch.cat([span_ind_mask,
                                             span_ind_mask.new_zeros((batch_size, left_mask_num_max))], -1)
            # SHAPE: (batch_size, num_spans, left_mask_num_max)
            span_ind_child = span_ind_child.unsqueeze(1).masked_select(left_mask).view(
                batch_size, num_spans, left_mask_num_max)
            span_ind_child_mask = span_ind_child_mask.unsqueeze(1).masked_select(left_mask).view(
                batch_size, num_spans, left_mask_num_max)

            # concat with parent ind
            span_pairs = torch.stack([span_ind.unsqueeze(2).repeat(1, 1, left_mask_num_max),
                                      span_ind_child], -1)
            span_pair_mask = torch.stack([span_ind_mask.unsqueeze(2).repeat(1, 1, left_mask_num_max),
                                          span_ind_child_mask], -1) > 0
            # SHAPE: (batch_size, num_spans * left_mask_num_max, 2)
            span_pairs = span_pairs.view(-1, num_spans * left_mask_num_max, 2)
            # SHAPE: (batch_size, num_spans * left_mask_num_max)
            span_pair_mask = span_pair_mask.view(-1, num_spans * left_mask_num_max, 2).all(-1).float()

            # TODO: Because of span_ind_mask, the result might not have left_size spans.
            #   This problem does not exist when the spans are all located at the top of the tensor
            return span_pairs, span_pair_mask, (num_spans, left_mask_num_max)

        if method == 'gold_predicate':
            _, num_spans2 = start_span_ind.size()
            # default: compose num_spans2 * num_spans pairs
            span_pairs = torch.stack([start_span_ind.unsqueeze(2).repeat(1, 1, num_spans),
                                      span_ind.unsqueeze(1).repeat(1, num_spans2, 1)], -1)
            span_pair_mask = torch.stack([start_span_ind_mask.unsqueeze(2).repeat(1, 1, num_spans),
                                          span_ind_mask.unsqueeze(1).repeat(1, num_spans2, 1)], -1)
            # SHAPE: (batch_size, num_spans2 * num_spans, 2)
            span_pairs = span_pairs.view(-1, num_spans2 * num_spans, 2)
            # SHAPE: (batch_size, num_spans * num_spans)
            span_pair_mask = span_pair_mask.view(-1, num_spans2 * num_spans, 2).all(-1).float()
            return span_pairs, span_pair_mask, (num_spans2, num_spans)

        # default: compose num_spans * num_spans pairs
        span_pairs = torch.stack([span_ind.unsqueeze(2).repeat(1, 1, num_spans),
                                  span_ind.unsqueeze(1).repeat(1, num_spans, 1)], -1)
        span_pair_mask = torch.stack([span_ind_mask.unsqueeze(2).repeat(1, 1, num_spans),
                                      span_ind_mask.unsqueeze(1).repeat(1, num_spans, 1)], -1)
        # SHAPE: (batch_size, num_spans * num_spans, 2)
        span_pairs = span_pairs.view(-1, num_spans * num_spans, 2)
        # SHAPE: (batch_size, num_spans * num_spans)
        span_pair_mask = span_pair_mask.view(-1, num_spans * num_spans, 2).all(-1).float()
        return span_pairs, span_pair_mask, (num_spans, num_spans)


    def extenral_to_internal(self,
                             span_ind: torch.LongTensor,  # SHAPE: (batch_size, num_spans)
                             total_num_spans: int,
                             ) -> Callable:  # SHAPE: (batch_size, total_num_spans)
        batch_size, num_spans = span_ind.size()
        # SHAPE: (batch_size, total_num_spans)
        converter = span_ind.new_zeros((batch_size, total_num_spans))
        new_ind = torch.arange(num_spans, device=span_ind.device).unsqueeze(0).repeat(batch_size, 1)
        # SHAPE: (batch_size, total_num_spans)
        converter.scatter_(-1, span_ind, new_ind)
        def converter_(ind):
            flat_ind = util.flatten_and_batch_shift_indices(ind, total_num_spans)
            new_ind = util.batched_index_select(converter.unsqueeze(-1), ind, flat_ind).squeeze(-1)
            return new_ind  # the same shape as ind
        return converter_


    def get_parent(self,
                   prob: torch.FloatTensor,  # SHAPE: (batch_size, num_span1, num_span2, num_class)
                   mask: torch.FloatTensor,  # SHAPE: (batch_size, num_span1, num_span2)
                   label: int,
                   ) -> torch.LongTensor:  # SHAPE: (batch_size, num_span1)
        ''' for each span find a parent span with maximum prob (-1 when parent does not exist) '''
        score, pred = prob.max(-1)
        mask = mask * pred.eq(label).float()
        has_parent = (mask.sum(-1) > 0).long()
        score = score + mask.log()
        # SHAPE: (batch_size, num_span1)
        pred = score.max(-1)[1]
        pred = pred * has_parent - (1 - has_parent)
        return pred


    def all_vs_one_loss(self,
                        logits: torch.FloatTensor,  # SHAPE: (batch_size, num_spans1, num_spans2, num_class)
                        mask: torch.FloatTensor,  # SHAPE: (batch_size, num_spans1, num_spans2)
                        label: torch.FloatTensor,  # SHAPE: (batch_size, num_spans1, num_spans2)
                        task_name: str,
                        major_label: str,
                        add_dummy: bool = True
                        ):
        batch_size, num_spans1, num_spans2, _ = logits.size()
        major_label = self.vocab.get_token_index(major_label, '{}_span_pair_labels'.format(task_name))
        logits = logits[:, :, :, major_label]
        label = label.eq(major_label).float() * mask  # must use mask to avoid fake label

        # add dummy class
        if add_dummy:
            logits = torch.cat([logits, logits.new_zeros(batch_size, num_spans1, 1)], -1)
            mask = torch.cat([mask, mask.new_ones(batch_size, num_spans1, 1)], -1)
            no_label = label.sum(-1).eq(0).float()
            label = torch.cat([label, no_label.unsqueeze(-1)], -1)

        # mask out irrelevent positions
        logits = logits + mask.log()
        # loss
        # SHAPE: (batch_size, num_spans1, num_spans2 + 1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # SHAPE: (batch_size, num_spans1)
        nll = -logsumexp(log_probs + label.log(), -1)
        # TODO: only use batch_size?
        #loss = nll.sum() / (mask.sum(-1) > 0).sum().float()
        loss = nll.sum() / batch_size
        return loss


    def all_max_mask(self,
                     prob: torch.FloatTensor,  # SHAPE: (batch_size, num_spans1, num_spans2, num_class)
                     mask: torch.FloatTensor,  # SHAPE: (batch_size, num_spans1, num_spans2)
                     label: torch.FloatTensor,  # SHAPE: (batch_size, num_spans1, num_spans2)
                     task_name: str,
                     major_label: str):
        batch_size, num_spans1, num_spans2, _ = prob.size()

        # get mask for major_label
        major_label = self.vocab.get_token_index(major_label, '{}_span_pair_labels'.format(task_name))
        label = label.eq(major_label).float() * mask

        # find the one with maximal prob
        prob = prob[:, :, :, major_label]
        prob = prob + label.log()
        # SHAPE: (batch_size, num_spans1)
        max_ind = prob.max(-1)[1]
        # SHAPE: (batch_size, num_spans1, num_span2)
        max_mask = torch.ones_like(label)
        max_mask.scatter_(-1, max_ind.unsqueeze(-1), 0)

        # mask without max
        # SHAPE: (batch_size, num_spans1, num_span2)
        no_max_mask = max_mask * label
        no_max_mask = (1 - no_max_mask) * mask

        return no_max_mask


    def get_num_spans_to_keep(self,
                              task_name: str,
                              seq_len: int,
                              max_value: int) -> int:
        spw = self._spans_per_word[task_name]
        if type(spw) is float:
            num_spans_to_keep = max(min(int(math.floor(self._spans_per_word[task_name] * seq_len)), max_value), 1)
        elif type(spw) is int:
            num_spans_to_keep = max(min(spw, max_value), 1)
        else:
            raise ValueError
        return num_spans_to_keep


def sequence_cross_entropy_with_logits(logits: torch.FloatTensor,
                                       targets: torch.LongTensor,
                                       weights: torch.FloatTensor,
                                       average: str = "batch",
                                       label_smoothing: float = None) -> torch.FloatTensor:
    if average not in {None, "token", "batch", "sum", "batch_sum"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', 'batch', 'sum', or 'batch_sum'")

    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights.float()

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights.sum().float() + 1e-13)
    elif average == "sum":
        return negative_log_likelihood.sum()
    elif average == "batch_sum":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        return per_batch_loss.sum()
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        return per_batch_loss


def pairwise_cosine_similarity(emb: torch.FloatTensor  # SHAPE: (batch_size, seq_len, emb_size)
                               ) -> torch.FloatTensor:
    # SHAPE: (batch_size, seq_len, emb_size)
    emb = emb / (emb.norm(dim=-1).unsqueeze(-1) + 1e-10)
    # SHAPE: (batch_size, seq_len, seq_len)
    cos = (emb.unsqueeze(2) * emb.unsqueeze(1)).sum(-1)
    return cos


def dump_similarity(sim: torch.FloatTensor,  # SHAPE: (batch_size, seq_len, seq_len)
                    sim_len: torch.LongTensor,  # SHAPE: (batch_size)
                    filename: str):
    with open(filename, 'a') as fout:
        for i in range(sim.size(0)):
            l = sim_len[i].item()
            s = sim[i][:l, :l].contiguous().view(-1).cpu().numpy()
            s = ' '.join(format(x, '.3f') for x in s)
            fout.write(s + '\n')
